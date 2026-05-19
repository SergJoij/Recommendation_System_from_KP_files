from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from sys import exit as sys_exit

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
import pprint

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "main_2.json"
REVIEWS_PATH = BASE_DIR / "reviews_3.json"
INTERACTIONS_PATH = BASE_DIR / "reviews_3_CF.json"

st.set_page_config(
    page_title="Обзорная информационная система",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    .hero {
        background:
            radial-gradient(circle at top left, rgba(255, 190, 92, 0.28), transparent 38%),
            linear-gradient(135deg, #1f2430 0%, #10151d 45%, #6e1f15 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        color: #f6efe4;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.18);
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        line-height: 1.15;
    }
    .hero p {
        margin: 0.25rem 0;
        font-size: 1rem;
        color: rgba(246, 239, 228, 0.9);
    }
    .mini-note {
        color: #8a96a8;
        font-size: 0.95rem;
    }
    .section-card {
        border: 1px solid rgba(20, 28, 45, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96));
    }
    .film-card {
        border: 1px solid rgba(20, 28, 45, 0.08);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        background: white;
        min-height: 150px;
    }
    .pill {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        margin: 0.08rem 0.18rem 0.08rem 0;
        border-radius: 999px;
        background: #f1e6d2;
        color: #7a4d0b;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def print_info(variable, type_=True) -> None:
    print(f"{variable=}\n{type(variable)}") if type_ else print(f"{variable=}")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


@st.cache_data(show_spinner=False)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies_raw = load_json(MOVIES_PATH)
    reviews_raw = load_json(REVIEWS_PATH)
    interactions_raw = load_json(INTERACTIONS_PATH)

    movies = pd.DataFrame(movies_raw).copy()
    reviews = pd.DataFrame(reviews_raw).copy()
    interactions = pd.DataFrame(interactions_raw).copy()

    movies["display_name"] = movies["name"].fillna("").replace("", np.nan)
    movies["display_name"] = movies["display_name"].fillna(movies["alternativeName"])
    movies["display_name"] = movies["display_name"].fillna("Без названия")
    movies["poster_preview"] = movies["poster"].apply(
        lambda value: (value or {}).get("previewUrl") if isinstance(value, dict) else None
    )
    movies["poster_url"] = movies["poster"].apply(
        lambda value: (value or {}).get("url") if isinstance(value, dict) else None
    )
    movies["genres"] = movies["genres"].apply(lambda value: value if isinstance(value, list) else [])
    movies["countries"] = movies["countries"].apply(lambda value: value if isinstance(value, list) else [])
    movies["genres_str"] = movies["genres"].apply(lambda value: ", ".join(value) if value else "не указаны")
    movies["countries_str"] = movies["countries"].apply(
        lambda value: ", ".join(value) if value else "не указаны"
    )
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int64")
    movies["kp_rating"] = pd.to_numeric(movies["kp_rating"], errors="coerce")
    movies["movie_length"] = pd.to_numeric(movies["movieLength"], errors="coerce")
    movies["total_series_length"] = pd.to_numeric(movies["totalSeriesLength"], errors="coerce")
    movies["series_length"] = pd.to_numeric(movies["seriesLength"], errors="coerce")
    movies["duration_minutes"] = (
        movies["movie_length"]
            .fillna(movies["total_series_length"])
            .fillna(movies["series_length"])
    )
    type_map = {
        "movie": "Фильм",
        "tv-series": "Сериал",
        "cartoon": "Мультфильм",
        "anime": "Аниме",
        "animated-series": "Мультсериал",
    }
    movies["type_label"] = movies["type"].map(type_map).fillna(movies["type"].fillna("Не указано"))

    reviews["review"] = reviews["review"].fillna("")
    reviews["title"] = reviews["title"].fillna("")
    reviews["createdAt"] = pd.to_datetime(reviews["createdAt"], errors="coerce")
    reviews["updatedAt"] = pd.to_datetime(reviews["updatedAt"], errors="coerce")
    reviews["reviewLikes"] = pd.to_numeric(reviews["reviewLikes"], errors="coerce").fillna(0)
    reviews["reviewDislikes"] = pd.to_numeric(reviews["reviewDislikes"], errors="coerce").fillna(0)
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce").fillna(0).astype(int)
    reviews["review_words"] = reviews["review"].str.split().str.len().fillna(0).astype(int)
    reviews["review_chars"] = reviews["review"].str.len().fillna(0).astype(int)
    reviews["binary_label"] = reviews["rating"].map({1: "Рекомендует", 0: "Не рекомендует"})

    interactions["authorId"] = pd.to_numeric(interactions["authorId"], errors="coerce").astype("Int64")
    interactions["movieId"] = pd.to_numeric(interactions["movieId"], errors="coerce").astype("Int64")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce").fillna(0).astype(int)
    interactions = interactions.dropna(subset=["authorId", "movieId"]).copy()
    interactions["authorId"] = interactions["authorId"].astype(int)
    interactions["movieId"] = interactions["movieId"].astype(int)

    movie_short = movies[
        [
            "id",
            "display_name",
            "year",
            "genres",
            "genres_str",
            "countries_str",
            "kp_rating",
            "poster_preview",
            "poster_url",
            "description",
            "shortDescription",
            "type_label",
            "duration_minutes",
        ]
    ].copy()
    reviews = reviews.merge(movie_short, left_on="movieId", right_on="id", how="left")
    interactions = interactions.merge(
        movie_short[
            ["id", "display_name", "genres", "genres_str", "year", "kp_rating", "poster_preview", "type_label"]
        ],
        left_on="movieId",
        right_on="id",
        how="left",
    )

    return movies, reviews, interactions


def build_stats(interactions: pd.DataFrame, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie_stats = (
        interactions.groupby("movieId")
            .agg(
            interactions_count=("rating", "size"),
            recommendation_rate=("rating", "mean"),
        )
            .reset_index()
    )
    user_stats = (
        interactions.groupby("authorId")
            .agg(
            total_reviews=("rating", "size"),
            positive_reviews=("rating", "sum"),
        )
            .reset_index()
    )
    user_stats["positive_share"] = user_stats["positive_reviews"] / user_stats["total_reviews"]

    if "display_name" in reviews.columns:
        review_counts = reviews.groupby(["movieId", "display_name"]).size().reset_index(name="reviews_count")
        movie_stats = movie_stats.merge(review_counts, on="movieId", how="left")

    return movie_stats, user_stats


def build_recommendation_assets(interactions: pd.DataFrame) -> dict:
    positive = interactions[interactions["rating"] == 1].copy()  # все позитивные отзывы
    # user_positive_movies: dict, ключи - id пользователя, значения - id фильмов, оцененных положительно
    user_positive_movies = positive.groupby("authorId")["movieId"].apply(set).to_dict()
    # movie_positive_users: dict, ключи - id фильма, значения - id пользователей, оценивших этот фильм положительно
    movie_positive_users = positive.groupby("movieId")["authorId"].apply(set).to_dict()
    # user_all_movies: dict, ключи - id пользователя, значения - id всех фильмов, им оцениваемых.
    user_all_movies = interactions.groupby("authorId")["movieId"].apply(set).to_dict()
    # movie_positive_rate: dict, ключи - id фильма, значения - средний рейтинг по оценке пользователей
    movie_positive_rate = interactions.groupby("movieId")["rating"].mean().to_dict()
    # movie_positive_rate: dict, ключи - id фильма, значения - количество оценок
    movie_interactions_count = interactions.groupby("movieId").size().to_dict()
    # max_movie_count: int, максимальное количество оценок для фильма (сам фильм не записыввется здесь)
    max_movie_count = max(movie_interactions_count.values()) if movie_interactions_count else 1
    return {
        "user_positive_movies": user_positive_movies,
        "movie_positive_users": movie_positive_users,
        "user_all_movies": user_all_movies,
        "movie_positive_rate": movie_positive_rate,
        "movie_interactions_count": movie_interactions_count,
        "max_movie_count": max_movie_count,
    }


def movie_label(movie_id: int, movies_indexed: pd.DataFrame, add_id=False) -> str:
    if movie_id in movies_indexed.index:
        row = movies_indexed.loc[movie_id]
        year = f" ({int(row['year'])})" if pd.notna(row["year"]) else ""
        return f"{row['display_name']}{year}" if not add_id else f"{row['display_name']}{year} - {movie_id}"
    return f"movieId={movie_id}"


def name_from_movie_label(movie_label_: str) -> str:
    split = movie_label_.rstrip().split(" ")[0:-1]
    result = " ".join(split)
    return result


def year_from_movie_label(movie_label_: str) -> int:
    split = movie_label_.rstrip().split(" ")
    result = int(split[-1][1:-1])
    return result


def name_year_from_movie_label(movie_label_: str) -> tuple[str, int]:
    split = movie_label_.rstrip().split(" ")
    year = int(split[-1][1:-1])
    name = " ".join(split[0:-1])
    return name, year


def id_from_name_year(movie_name: str, movie_year: int) -> int:
    """Возвращает id фильма исходя из его названия и года"""
    result = st.session_state.movies["id"][
        (st.session_state.movies["name"] == movie_name) & (st.session_state.movies["year"] == movie_year)].values[0]
    return result


def id_from_movie_label(movie_label_: str, add_id=False, get_name_year=False) -> tuple | int:
    """Возвращает id фильма исходя из его названия и года"""
    movie_name, movie_year = name_year_from_movie_label(movie_label_)
    if not add_id:
        result = st.session_state.movies["id"][
            (st.session_state.movies["name"] == movie_name) & (st.session_state.movies["year"] == movie_year)].values[0]
    else:
        result = int(movie_label_.rstrip().split(" ")[-1])
    return (result, movie_name, movie_year) if get_name_year else result


def build_top_movie_pool(interactions: pd.DataFrame, movies_indexed: pd.DataFrame) -> list[int]:
    top_items = (
        interactions.groupby("movieId")
            .agg(count=("rating", "size"), recommend_share=("rating", "mean"))
            .sort_values(["count", "recommend_share"], ascending=[False, False])
            .head(300)
            .index.tolist()
    )
    return [movie_id for movie_id in top_items if movie_id in movies_indexed.index]


def score_unseen_movie(author_id: int, movie_id: int, assets: dict) -> dict:
    """ Для каждого фильма из списка и выбранного. Выводит результат для фильма.
    assets - это список списков, состоит из ключей:
       'user_positive_movies' - ключ id пользователя, значение - сет id всех понравившихся ему фильмов;
       'movie_positive_users' - ключ id фильма, значения - сет id всех пользователей, которым понравивился фильм,
          показатель обратный для user_positive_movies;
       'user_all_movies' - ключ id пользователя, значения - сет id всех оцененных им фильмов;
       'movie_positive_rate' - ключ id фильма, значение - средний рейтинг положительых оценок пользователей
          (кол-во полож / все), высчитан с помощью mean, связан с movie_positive_users;
       'movie_interactions_count' - ключ id фильма, значение - количество оценок пользователей,
       'max_movie_count' - int, максимальное количество всех оценок фильма (сам фильм не записыввется здесь).
    """
    # print("assets")
    # print(assets['max_movie_count'])
    # sys_exit(0)

    liked_movies = assets["user_positive_movies"].get(author_id, set())  # фильмы, которым польз. пост. позитив. оценку
    seen_movies = assets["user_all_movies"].get(author_id, set())  # все фильмы, которые оценивал наш пользователь
    positive_rate = float(assets["movie_positive_rate"].get(movie_id, 0.5))  # средняя оценка нашего фильма

    # объем: количество оценок этого фильма деленное на кол-во оценок для фильма, у которого их количество - макс.
    volume = math.log1p(assets["movie_interactions_count"].get(movie_id, 0)) / math.log1p(assets["max_movie_count"])

    if movie_id in seen_movies:  # если уже есть оценка от нашего пользователя
        print("movie_id in seen_movies")
        actual_rating = 1 if movie_id in liked_movies else 0
        probability = 0.92 if actual_rating == 1 else 0.08
        return {
            "probability": probability,
            "neighbor_score": float(actual_rating),
            "positive_rate": positive_rate,
            "volume_score": volume,
            "support_users": len(assets["movie_positive_users"].get(movie_id, set())),
            "already_seen": True,
            "actual_rating": actual_rating,
        }

    # далее - если ещё нет оценки от нашего пользователя
    # пользователи, которые оценили наш фильм положительно:
    candidate_users = assets["movie_positive_users"].get(movie_id, set())
    # если наш фильм положительно не оценивал любой другой пользователь (и наш, но они и так отсеялся ранее)
    if not liked_movies or not candidate_users:
        print("not liked_movies or not candidate_users")
        raw_score = 2.1 * (positive_rate - 0.5) + 0.6 * volume  # идёт сырая оценка вероятности
        # probability = 1 / (1 + math.exp(-raw_score))
        probability = st.session_state.model.predict_proba(
            pd.DataFrame(data={'authorId': [author_id], 'movieId': [movie_id]}, index=[0, 1]))[0][1]
        return {
            "probability": probability,
            "neighbor_score": 0.0,
            "positive_rate": positive_rate,
            "volume_score": volume,
            "support_users": len(candidate_users),
            "already_seen": False,
            "actual_rating": None,
        }

    # если кто-то другой из пользователей оценил положительно наш фильм
    # print("OTHER")
    similarities: list[float] = []   # схожести
    for other_user in candidate_users:  # по каждому кандидату
        if other_user == author_id:
            continue
        other_likes = assets["user_positive_movies"].get(other_user, set())
        if not other_likes:
            continue
        overlap = len(liked_movies & other_likes)
        if overlap == 0:
            continue
        similarity = overlap / math.sqrt(len(liked_movies) * len(other_likes))
        similarities.append(similarity)

    similarities = sorted(similarities, reverse=True)[:25]
    neighbor_score = float(sum(similarities) / len(similarities)) if similarities else 0.0

    # raw_score = 3.4 * neighbor_score + 1.8 * (positive_rate - 0.5) + 0.8 * volume - 0.15
    # probability = 1 / (1 + math.exp(-raw_score))
    probability = st.session_state.model.predict_proba(pd.DataFrame(data={'authorId': [author_id], 'movieId': [movie_id]}, index=[0, 1]))
    return {
        "probability": probability[0][1],       # вероятность рекомендации, т.е. попадания в класс 1
        "neighbor_score": neighbor_score,
        "positive_rate": positive_rate,
        "volume_score": volume,
        "support_users": len(candidate_users),  # количество пользователей, положительно оценивших фильм
        "already_seen": False,
        "actual_rating": None,
    }


def recommend_for_user(author_id: int, assets: dict, interactions: pd.DataFrame) -> pd.DataFrame:
    """ Возвращает df фильмов-кандидатов с полями "movieId", "final_score", "probability" и т.д., отсортированных по
    "final_score" - более сложной вероятности, "probability" - обычной верятности - и "support_users" от высокого.

    assets - это список списков, состоит из ключей:
       'user_positive_movies' - ключ id пользователя, значение - сет id всех понравившихся ему фильмов;
       'movie_positive_users' - ключ id фильма, значения - сет id всех пользователей, которым понравивился фильм,
          обратный для user_positive_movies;
       'user_all_movies' - ключ id пользователя, значения - сет id всех оцененных им фильмов;
       'movie_positive_rate' - ключ id фильма, значение - средний рейтинг положительых оценок пользователей
          (кол-вополож / все), высчитан с помощью mean, связан с movie_positive_users;
       'movie_interactions_count' - ключ id фильма, значение - количество оценок пользователей;
       'max_movie_count' - int, максимальное количество всех оценок фильма (сам фильм не записыввется здесь).
    """
    liked_movies = assets["user_positive_movies"].get(author_id, set())  # set понравившихся фильмов
    seen_movies = assets["user_all_movies"].get(author_id, set())  # set всех оцененных фильмов

    overlap_scores: dict[int, float] = defaultdict(float)  # словарь - id фильма: сходство
    if liked_movies:  # если set понравившихся фильмов не пустой и вообще существует (т.е. у польз. есть фильмы, оц. 1)
        # Counter - коллекция счётчик, подкласс словаря,, где ключ - элемент, значение - количество эл-тов в контейнере
        # форма записи ниже инициализирует пустой Counter - переменная neighbor_overlap, аннотация Counter,
        #    объект Counter()
        neighbor_overlap: Counter = Counter()
        for movie_id in liked_movies:  # по всем фильмам, понравившимся нашему пользователю
            # по всем пользователям, которые оценили понравившимся нашему пользователю фильм положительно:
            for other_user in assets["movie_positive_users"].get(movie_id, set()):
                if other_user != author_id:  # если пользователь - не автор
                    # добавляем пользователя в счётчик (важно количество его присутствий)
                    neighbor_overlap[other_user] += 1

        # по 80 пользователям, интересы которых в оценке фильмов наиболее близки к нашим
        for other_user, overlap in neighbor_overlap.most_common(80):  # other_user - id польз., overlap - кол-во оценок
            # сет id всех понравившихся other_user фильмов
            other_likes = assets["user_positive_movies"].get(other_user, set())
            if not other_likes:  # если таких нет (вряд ли такое может быть, но на всякий)
                continue
            # схожесть того пользователя с нашим:
            #    кол-во попаданий / sqrt(кол-во фильмов, понр нашему пользователю * кол-во фильмов, понр. other_user)
            similarity = overlap / math.sqrt(len(liked_movies) * len(other_likes))
            for movie_id in other_likes:
                if movie_id in seen_movies:
                    continue
                overlap_scores[movie_id] += similarity  # добавляем в словарь

    # получаем overlap_scores - словарь фильмов, которые понравились другим пользователям, положительный оценки которых
    #    схожи с положительными оценками нашего пользователя, значение словаря - сходство
    candidate_pool = set(overlap_scores)  # сет фильмов кандидатов для нашего пользователя
    if not candidate_pool:  # если set пустой или его не существует, то берём самые высокие по рейтингу фильмы
        fallback = (
            interactions.groupby("movieId")
                .agg(count=("rating", "size"), recommend_share=("rating", "mean"))
                .query("count >= 8")
                .sort_values(["recommend_share", "count"], ascending=[False, False])
                .head(100)
                .index.tolist()
        )
        candidate_pool = {movie_id for movie_id in fallback if movie_id not in seen_movies}

    # теперь вероятность и т.д.
    rows = []
    # значение сходства наиболее подходященр фильма-кандидата
    max_overlap = max(overlap_scores.values()) if overlap_scores else 1.0
    for movie_id in candidate_pool:  # по фильмам кандидатам
        score = score_unseen_movie(author_id, movie_id, assets)  # вероятность и т.д.
        overlap_component = overlap_scores.get(movie_id, 0.0) / max_overlap if max_overlap else 0.0
        # более сложная вероятность?
        final_score = 0.65 * score["probability"] + 0.2 * overlap_component + 0.15 * score["positive_rate"]
        rows.append(
            {
                "movieId": movie_id,
                "final_score": final_score,
                "probability": score["probability"],
                "overlap_component": overlap_component,
                "recommend_share": score["positive_rate"],
                "support_users": score["support_users"],
            }
        )

    # dataframe из кандидатов: id фильма, final_score, вероятность и другие параметры
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # print(f"result\n{result}")
    return result.sort_values(["final_score", "probability", "support_users"], ascending=False).reset_index(drop=True)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_movie_summary(movie_row: pd.Series) -> None:
    print("render_movie_summary")
    poster = movie_row.get("poster_preview") or movie_row.get("poster_url")
    if poster:
        poster = poster.replace("image.openmoviedb.com", "avatars.mds.yandex.net", 1).replace("kinopoisk-images", "get-kinopoisk-image", 1)
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.caption("Постер отсутствует")
    with right:
        st.subheader(movie_row["display_name"])
        meta_parts = [
            str(int(movie_row["year"])) if pd.notna(movie_row["year"]) else None,
            movie_row.get("type_label"),
            f"КП: {movie_row['kp_rating']:.2f}" if pd.notna(movie_row["kp_rating"]) else None,
            f"{int(movie_row['duration_minutes'])} мин." if pd.notna(movie_row["duration_minutes"]) else None,
        ]
        st.write(" • ".join([part for part in meta_parts if part]))
        st.write(movie_row.get("shortDescription") or movie_row.get("description") or "Описание отсутствует")
        pills = "".join([f"<span class='pill'>{genre}</span>" for genre in movie_row.get("genres", [])[:6]])
        if pills:
            st.markdown(pills, unsafe_allow_html=True)
        st.caption(f"Страны: {movie_row.get('countries_str', 'не указаны')}")


def render_recommendation_cards(recommendations: pd.DataFrame, movies_indexed: pd.DataFrame, top_n: int = 6) -> None:
    if recommendations.empty:
        st.warning("Для выбранного пользователя пока не удалось собрать рекомендации.")
        return

    top_rows = recommendations.head(top_n).copy()   # dataframe, топ самых рекомендуемых пользователю фильмов
    # print_info(top_rows)
    columns = st.columns(3, gap="large")
    for idx, (_, row) in enumerate(top_rows.iterrows()):
        movie_id = int(row["movieId"])
        movie = movies_indexed.loc[movie_id]
        with columns[idx % 3]:
            poster = movie.get("poster_preview") or movie.get("poster_url")
            if poster:
                st.image(poster, use_container_width=True)
            st.markdown(
                f"""
                <div class="film-card">
                    <strong>{movie['display_name']}</strong><br>
                    <span class="mini-note">{movie.get('type_label', '')}</span><br><br>
                    Вероятность рекомендации: <strong>{format_percent(row['probability'])}</strong><br>
                    Доля позитивных отзывов: <strong>{format_percent(row['recommend_share'])}</strong><br>
                    Поддержка похожих пользователей: <strong>{int(row['support_users'])}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_user_history(author_id: int, interactions: pd.DataFrame, movies_indexed: pd.DataFrame) -> pd.DataFrame:
    history = interactions[interactions["authorId"] == author_id].copy()
    if history.empty:
        return history
    history["movie_label"] = history["movieId"].apply(lambda movie_id: movie_label(int(movie_id), movies_indexed))
    return history.sort_values(["rating", "kp_rating"], ascending=[False, False])


def get_new_index(data_: pd.DataFrame) -> int:
    """Автоматически подбирает новый id для data_"""
    values = data_["authorId"].values
    min_ind = min(values)
    result = None

    if min_ind > 1:
        result = min_ind - 1
    else:
        max_ind = max(values)
        if len(str(max_ind + 1)) == len(str(max_ind)):  # если при прибавлении кол-во разрядов не увеличится
            result = max_ind + 1
        else:
            for el in range(min_ind + 1, max_ind):
                if el not in values:
                    result = el
                    break
    return result if result is not None else max(values) + 1


def get_new_indexes(values: np.ndarray | list | tuple, number: int, permission=True) -> list[int]:
    """ Автоматически подбирает список новых id для data_
    number - размер возвращаемого списка или количество нужных id
    permission - разрешение на превышение количества разрадов
    """
    min_ind, max_ind = min(values), max(values)
    result = []
    stop_cycle = False
    for _ in range(number):
        if min_ind > 1:
            result.append(min_ind - 1)
            min_ind = min_ind - 1
        else:
            if len(str(max_ind + 1)) == len(str(max_ind)) or permission:
                result.append(max_ind + 1)
                max_ind = max_ind + 1
            else:
                for el in range(min_ind + 1, max_ind):  # вот этот цикл будет добавлять элементы до конца
                    if stop_cycle:
                        break
                    if el not in values and el not in result:
                        result.append(el)
                        if result.__len__() == number:
                            stop_cycle = True
    return result


def render_overview(user_stats: pd.DataFrame) -> None:
    print("render_overview")
    interactions = st.session_state.interactions
    reviews = st.session_state.reviews
    movies = st.session_state.movies

    density = len(interactions) / (interactions["authorId"].nunique() * interactions["movieId"].nunique())
    long_reviews_share = (reviews["review_words"] >= 300).mean()

    st.markdown(
        f"""
        <div class="hero">
            <h1>Рекомендательная система для фильмов на основе данных Кинопоиска</h1>
            <p>Streamlit-витрина для демонстрации обзорной информационной системы: данные, исследование, сравнение моделей и live-сценарий рекомендаций.</p>
            <p>Локальные источники: <strong>{MOVIES_PATH.name}</strong>, <strong>{REVIEWS_PATH.name}</strong>, 
            <strong>{INTERACTIONS_PATH.name}</strong>, <strong>catboost_bin.json</strong>.</p>
            <p> База данных была построена на информации о фильмах, снятых в России.<p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Фильмов", f"{movies['id'].nunique():,}".replace(",", " "))
    metric_2.metric("Отзывов", f"{len(reviews):,}".replace(",", " "))
    metric_3.metric("Пользователей", f"{interactions['authorId'].nunique():,}".replace(",", " "))
    metric_4.metric("Плотность матрицы", f"{density:.4%}")
    metric_5.metric("Длинные рецензии (от 300 слов)", f"{long_reviews_share:.1%}")

    # left, right = st.columns([1.25, 1], gap="large")
    # with left:
    #     st.markdown("### Что информационная система показывает на защите")
    #     st.markdown(
    #         """
    #         - исследовательский контур диплома: от сбора JSON-данных до сравнения моделей;
    #         - аналитический профиль датасета: разреженность, баланс классов, активность пользователей;
    #         - визуальное представление фильмов и отзывов с постерами, жанрами и текстами рецензий;
    #         - live-демо рекомендаций для выбранного пользователя на тех же локальных данных.
    #         """
    #     )
    # with right:
    #     st.markdown("### Почему это хороший формат для демонстрации")
    #     st.markdown(
    #         """
    #         - защита не зависит от внешнего API и интернета;
    #         - можно быстро переключаться между аналитикой и кейсами пользователя;
    #         - интерфейс показывает не только метрики, но и содержательный результат системы.
    #         """
    #     )

    top_active = user_stats.sort_values(["total_reviews", "positive_share"], ascending=[False, False]).head(10)
    st.markdown("### Наиболее активные пользователи")
    st.dataframe(
        top_active.assign(
            positive_share=top_active["positive_share"].map(lambda value: f"{value:.1%}")
        ).rename(columns={"authorId": "Пользователь",
                          "total_reviews": "Количество рецензий",
                          "positive_reviews": "Количество положительных рецензий",
                          "positive_share": "Позитивная доля"}),
        hide_index=True,
        use_container_width=True,
        # "Фильм", "Прогноз", "Позитивная доля", "support_users"
    )


def render_data_tab(movie_stats: pd.DataFrame, movies_indexed: pd.DataFrame) -> None:
    reviews = st.session_state.reviews
    interactions = st.session_state.interactions
    movies = st.session_state.movies

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("### Количество данных для каждого класса отзывов")
        sentiment_counts = reviews["type"].fillna("Не указано").value_counts()
        st.bar_chart(sentiment_counts)

        st.markdown("### Бинарная целевая переменная")
        binary_counts = interactions["rating"].map({1: "Рекомендует", 0: "Не рекомендует"}).value_counts()
        st.bar_chart(binary_counts)

    with col_b:
        st.markdown("### Длина рецензий по числу слов")
        bins = pd.cut(
            reviews["review_words"],
            bins=[0, 100, 200, 300, 500, 800, 1200],
            labels=["0-100", "101-200", "201-300", "301-500", "501-800", "801-1200"],
            include_lowest=True,
        )
        st.bar_chart(bins.value_counts().sort_index())

        st.markdown("### Годы выпуска фильмов")
        years = movies["year"].dropna().astype(str).value_counts().sort_index()[1:-1]
        st.line_chart(years)

    st.markdown("### Наиболее обсуждаемые фильмы")
    top_movies = (
        movie_stats.sort_values(["interactions_count", "recommendation_rate"], ascending=[False, False])
            .head(15)
            .copy()
    )
    top_movies["Фильм"] = top_movies["movieId"].apply(lambda movie_id: movie_label(int(movie_id), movies_indexed))
    top_movies["Доля рекомендаций"] = top_movies["recommendation_rate"].map(lambda value: f"{value:.1%}")  # Series
    rest_ = pd.merge(top_movies, movies[["id", "kp_rating"]], left_on='movieId', right_on='id', how='inner')
    top_movies["Рейтинг"] = rest_["kp_rating"].values
    st.dataframe(
        top_movies[["Фильм", "interactions_count", "Доля рекомендаций", "Рейтинг"]].rename(
            columns={"interactions_count": "Количество отзывов"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    # st.markdown("### Примеры фильмов в витрине")
    # showcase = top_movies["movieId"].head(3).tolist()
    # cols = st.columns(3, gap="large")
    # for idx, movie_id in enumerate(showcase):
    #     if movie_id not in movies_indexed.index:
    #         continue
    #     with cols[idx]:
    #         movie = movies_indexed.loc[movie_id]
    #         poster = movie.get("poster_preview") or movie.get("poster_url")
    #         if poster:
    #             st.image(poster, use_container_width=True)
    #         st.markdown(
    #             f"""
    #             <div class="film-card">
    #                 <strong>{movie['display_name']}</strong><br>
    #                 <span class="mini-note">{movie.get('genres_str', '')}</span><br><br>
    #                 Рейтинг КП: <strong>{movie['kp_rating']:.2f}</strong><br>
    #                 Год: <strong>{int(movie['year']) if pd.notna(movie['year']) else '—'}</strong>
    #             </div>
    #             """,
    #             unsafe_allow_html=True,
    #         )


def render_models_tab() -> None:
    st.markdown("### Сводка лучших результатов из дипломной работы")
    # st.info(
    #     "В live-демо ниже используется локальная коллаборативная рекомендация на тех же данных. "
    #     "Отдельный сериализованный артефакт CatBoost в папке пока не сохранён, поэтому метрики CatBoost "
    #     "здесь показаны по материалам диплома."
    # )

    binary_results = pd.DataFrame(
        [
            {"Модель": "KNN", "F1 по классам [0, 1]": "0.58785249 0.73519164", "F1": 0.662, "Kappa": 0.331,
             "Комментарий": "лучший KNN"},
            {"Модель": "SVD", "F1 по классам [0, 1]": "0.59304905 0.71734292", "F1": 0.655, "Kappa": 0.311,
             "Комментарий": "лучший SVD"},
            {
                "Модель": "CatBoost",
                "F1 по классам [0, 1]": "0.65102421 0.68461797",
                "F1": 0.668,
                "Kappa": 0.340,
                "Комментарий": "лучшая итоговая модель",
            },
        ]
    )
    multiclass_results = pd.DataFrame(
        [
            {"Модель": "KNN", "F1 по классам [-1, 0, 1]": "0.1990172 0.32327167 0.56859362", "F1": 0.364,
             "Kappa": 0.152},
            {"Модель": "SVD (best F1)", "F1 по классам [-1, 0, 1]": "0.21583851 0.32197562 0.60095071", "F1": 0.380,
             "Kappa": 0.164},
            {"Модель": "SVD (best Kappa)", "F1 по классам [-1, 0, 1]": "0.17770035 0.33365354 0.60272509", "F1": 0.371,
             "Kappa": 0.168},
            {"Модель": "CatBoost", "F1 по классам [-1, 0, 1]": "0.52181003 0.34132581 0.63688368", "F1": 0.500,
             "Kappa": 0.280},
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Бинарная классификация")
        st.dataframe(binary_results, hide_index=True, use_container_width=True)
        # st.bar_chart(binary_results.set_index("Модель")[["F1", "Kappa"]])
    with right:
        st.markdown("#### Многоклассовая классификация")
        st.dataframe(multiclass_results, hide_index=True, use_container_width=True)
        # st.bar_chart(multiclass_results.set_index("Модель")[["F1", "Kappa"]])

    with left:
        st.bar_chart(binary_results.set_index("Модель")[["F1", "Kappa"]])
    with right:
        st.bar_chart(multiclass_results.set_index("Модель")[["F1", "Kappa"]])

    st.markdown("### Выводы")
    st.markdown(
        """
        - переход от трёх классов к двум классам заметно упрощает задачу и даёт прирост качества;
        - CatBoost показывает лучшие результаты `F1` и `Kappa`;
        """
    )
    # для демонстрации на защите разумно совмещать таблицу метрик из диплома и интерактивные рекомендации
    # по реальным данным.


def render_demo_tab(user_stats: pd.DataFrame, assets: dict) -> None:
    """
    assets - это список списков, состоит из ключей:
       'user_positive_movies' - ключ id пользователя, значение - сет id всех понравившихся ему фильмов;
       'movie_positive_users' - ключ id фильма, значения - сет id всех пользователей, которым понравивился фильм,
          обратный для user_positive_movies;
       'user_all_movies' - ключ id пользователя, значения - сет id всех оцененных им фильмов;
       'movie_positive_rate' - ключ id фильма, значение - средний рейтинг положительых оценок пользователей
          (кол-вополож / все), высчитан с помощью mean, связан с movie_positive_users;
       'movie_interactions_count' - ключ id фильма, значение - количество оценок пользователей;
       'max_movie_count' - int, максимальное количество всех оценок фильма (сам фильм не записыввется здесь).
    """
    def highlight_rows(row: pd.Series) -> list[str]:  # цвет строк
        number = int(row['Прогноз'].split(".")[0])  # целая часть процента, red_number = 100 - number
        green_number = int(number * 2.55)
        red_number = int((100 - number) * 2.55)
        result = f'background-color: rgba({red_number},{green_number},0,0.25)'
        return [result] * len(row)

    movies = st.session_state.movies
    interactions = st.session_state.interactions

    movies_indexed = movies.set_index("id")  # фильмы, только индекс df у них - id фильмов
    demo_users = (
        user_stats[user_stats["total_reviews"] >= 3]  # >= 5
            .sort_values(["total_reviews", "positive_share"], ascending=[False, False])
    )  # было ещё .head(400), но без него не виден новый пользователь (можно исправить сортировкой ?)

    def user_option_format(author_id: int) -> str:
        row = demo_users.loc[demo_users["authorId"] == author_id].iloc[0]
        return (
                f"{author_id} • {int(row['total_reviews'])} отзывов • " +
                f"{row['positive_share']:.0%} позитивных"
        )

    sidebar_left, sidebar_right = st.columns([1, 1.1], gap="large")
    with sidebar_left:
        selected_user = st.selectbox(
            "Выберите пользователя для демонстрации",
            options=demo_users["authorId"].astype(int).tolist(),
            index=0,
            format_func=user_option_format,
        )
    with sidebar_right:
        top_movie_pool = build_top_movie_pool(interactions, movies_indexed)
        selected_movie = st.selectbox(
            "Выберите фильм",
            options=top_movie_pool,
            index=0,
            format_func=lambda movie_id: movie_label(int(movie_id), movies_indexed),
        )

    history = get_user_history(int(selected_user), interactions, movies_indexed)
    recommendations = recommend_for_user(int(selected_user), assets, interactions)  # эти
    # print("Дойдёт ли")
    scored_movie = score_unseen_movie(int(selected_user), int(selected_movie), assets)  # эти

    stats_row = user_stats.loc[user_stats["authorId"] == selected_user].iloc[0]
    liked_count = len(assets["user_positive_movies"].get(int(selected_user), set()))
    seen_count = len(assets["user_all_movies"].get(int(selected_user), set()))

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Отзывов у пользователя", int(stats_row["total_reviews"]))
    metric_b.metric("Позитивных отзывов", liked_count)
    metric_c.metric("Оценено фильмов", seen_count)
    metric_d.metric("Доля рекомендаций", f"{stats_row['positive_share']:.1%}")

    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.markdown("### Прогноз по выбранному фильму")
        render_movie_summary(movies_indexed.loc[int(selected_movie)])
        badge = "Уже есть в истории пользователя" if scored_movie["already_seen"] else "Новый кандидат"
        st.success(
            f"{badge}. Вероятность рекомендации: {format_percent(scored_movie['probability'])}. "
            f"Позитивная доля по фильму: {format_percent(scored_movie['positive_rate'])}."
        )
        if scored_movie["already_seen"]:
            actual = "рекомендовал" if scored_movie["actual_rating"] == 1 else "не рекомендовал"
            st.caption(f"Пользователь уже {actual} этот фильм.")
        else:
            st.caption(
                f"Поддержка похожих пользователей: {scored_movie['support_users']}. "
                f"Сигнал соседей: {scored_movie['neighbor_score']:.3f}."
            )

    with right:
        st.markdown("### История пользователя")
        if history.empty:
            st.warning("История взаимодействий отсутствует.")
        else:
            display_history = history[
                ["movie_label", "rating", "kp_rating", "genres_str", "type_label"]
            ].rename(
                columns={
                    "movie_label": "Фильм",
                    "rating": "Метка",
                    "kp_rating": "КП",
                    "genres_str": "Жанры",
                    "type_label": "Тип",
                }
            )
            display_history["Метка"] = display_history["Метка"].map({1: "Рекомендует", 0: "Не рекомендует"})
            st.dataframe(display_history.head(12), hide_index=True, use_container_width=True)

    # st.markdown("### Рекомендации top-N")
    # render_recommendation_cards(recommendations, movies_indexed, top_n=6)  # почти всегда не работает, не обязательно
    if not recommendations.empty:
        st.markdown("### Таблица рекомендаций")
        table_place = st.empty()
        color_activate = st.toggle("Цвет таблицы", value=True)

        recommendation_table = recommendations.head(20).copy()
        recommendation_table["Фильм"] = recommendation_table["movieId"].apply(
            lambda movie_id: movie_label(int(movie_id), movies_indexed)
        )
        recommendation_table["Прогноз"] = recommendation_table["probability"].map(format_percent)
        recommendation_table["Позитивная доля"] = recommendation_table["recommend_share"].map(format_percent)
        if color_activate:
            with table_place:
                st.dataframe(
                        recommendation_table[
                            ["Фильм", "Прогноз", "Позитивная доля", "support_users"]
                        ].rename(columns={"support_users": "Поддержка"}).style.apply(highlight_rows, axis=1),
                        hide_index=True,
                        use_container_width=True,
                    )
        else:
            with table_place:
                st.dataframe(
                    recommendation_table[
                        ["Фильм", "Прогноз", "Позитивная доля", "support_users"]
                    ].rename(columns={"support_users": "Поддержка"}),
                    hide_index=True,
                    use_container_width=True,
                )


def render_reviews_tab() -> None:
    movies = st.session_state.movies
    reviews = st.session_state.reviews

    movies_indexed = movies.set_index("id")
    top_reviewed = (
        reviews.groupby("movieId")
            .size()
            .sort_values(ascending=False)
            .head(500)
            .index.tolist()
    )
    # top_reviewed = build_top_movie_pool(interactions, movies_indexed)
    top_reviewed = [movie_id for movie_id in top_reviewed if movie_id in movies_indexed.index]

    left, right = st.columns([1, 1], gap="large")
    with left:
        selected_movie = st.selectbox(
            "Фильм для разбора отзывов",
            options=top_reviewed,
            format_func=lambda movie_id: movie_label(int(movie_id), movies_indexed),
        )
    with right:
        sentiment_filter = st.multiselect(
            "Типы отзывов",
            options=sorted(reviews["type"].dropna().unique().tolist()),
            default=sorted(reviews["type"].dropna().unique().tolist()),
        )

    movie = movies_indexed.loc[int(selected_movie)]
    render_movie_summary(movie)

    current_reviews = reviews[reviews["movieId"] == int(selected_movie)].copy()
    if sentiment_filter:
        current_reviews = current_reviews[current_reviews["type"].isin(sentiment_filter)]

    review_metrics = st.columns(4)
    review_metrics[0].metric("Отзывов", len(current_reviews))
    review_metrics[1].metric("Средняя длина", f"{current_reviews['review_words'].mean():.0f} слов")
    review_metrics[2].metric("Средние лайки", f"{current_reviews['reviewLikes'].mean():.1f}")
    review_metrics[3].metric("Средние дизлайки", f"{current_reviews['reviewDislikes'].mean():.1f}")

    st.markdown("### Распределение отзывов по типам")
    st.bar_chart(current_reviews["type"].value_counts())

    st.markdown("### Примеры рецензий")
    sample_reviews = current_reviews.sort_values(["reviewLikes", "review_words"], ascending=[False, False]).head(8)
    for _, row in sample_reviews.iterrows():
        heading = row["title"].strip() or "Без заголовка"
        if row['reviewLikes'] is None or np.isnan(row['reviewLikes']):
            print("type", type(row['reviewLikes']))
            row['reviewLikes'] = 0
            row['reviewDislikes'] = 0
        meta = (
            f"{row['author']} • {row['type']} • "
            f"лайки: {int(row['reviewLikes'])} • дизлайки: {int(row['reviewDislikes'])}"
        )
        with st.expander(f"{heading}"):
            st.caption(meta)
            st.write(row["review"])


def get_page_table(df: pd.DataFrame, movies_list: list | tuple) -> st.data_editor:
    result_df = st.data_editor(
        df,
        column_config={
            "Фильм": st.column_config.SelectboxColumn(
                "Фильм",
                help="Выберите фильм",
                options=movies_list,  # variants_1  top_movie_pool
                required=True
            ),
            "Заголовок": st.column_config.TextColumn("Заголовок", disabled=False),
            "Отзыв": st.column_config.TextColumn("Отзыв", disabled=False),
            "Тип": st.column_config.SelectboxColumn(
                "Тип",
                options=["Нейтральный", "Позитивный", "Негативный"],
                required=True
            )
        },
        num_rows="dynamic",
        use_container_width=True,
        key="film_editor"
    )
    return result_df


def fields_from_id(list_of_id: list, fields: list[str], movies_indexed: pd.DataFrame) -> dict:
    result = dict.fromkeys(fields)
    for key in result.keys():
        result[key] = []
    for id_ in list_of_id:
        for field in fields:
            value = movies_indexed.loc[id_][field]
            result[field].append(value)
    return result


def additional_training(data: pd.DataFrame) -> None:
    if data['rating'].nunique() == 2:
        # дообучение модели
        # print("df", data[['authorId', 'movieId', 'rating']])
        new_model = CatBoostClassifier(
            iterations=90,  # Сколько дополнительных деревьев построить (обычно меньше, чем в базе = 100)
            learning_rate=0.09,  # Можно чуть уменьшить шаг, чтобы не "сломать" старые знания (было 0.1)
            depth=4,
            cat_features=['authorId', 'movieId'],
            random_seed=42,
            silent=True,
            eval_metric='AUC',
            custom_metric=['Precision', 'Recall', 'Kappa', 'F1'],
            auto_class_weights='Balanced',
            loss_function='Logloss'
        )
        print("new_model created")
        new_model.fit(
            data[['authorId', 'movieId']],
            y=data['rating'],
            init_model=st.session_state.model  # состояние старой модели
        )
        st.session_state.model = new_model
        del new_model
        print("Модель обучена на дополнительных данных!", st.session_state.model)
    else:
        print("Не хватает разнообразных данных для дообучения. Модель останется в прежнем виде!")


def update_reviews(df_original: pd.DataFrame, user_name: str, user_id: int, movies_indexed: pd.DataFrame):
    """Обновляет глобальные переменные сессии для использования в других вкладках далее"""
    if user_name is None or user_name.replace(" ", "") == "":  # нет или состоит из пробелов - пустое
        user_name = "NewUser2026"  # заглушка
    df = df_original.copy()

    # reviews
    id_x = get_new_indexes(st.session_state.reviews["id_x"].values, df.shape[0])  # id_x
    df["id_x"] = id_x

    df.rename(columns={'Заголовок': 'title', "Тип": 'type', 'Отзыв': 'review'}, inplace=True)

    movie_names = list(map(lambda movie: name_from_movie_label(movie), df["Фильм"].values))
    movie_years = list(map(lambda movie: year_from_movie_label(movie), df["Фильм"].values))

    # movie_ids = list(map(lambda movie: id_from_movie_label(movie), df["Фильм"].values))
    movie_ids = list(map(lambda movie: id_from_name_year(movie[0], movie[1]), zip(movie_names, movie_years)))
    df["id_y"] = movie_ids
    df["movieId"] = movie_ids

    df["author"] = user_name

    df["authorId"] = user_id
    df["authorId"] = df["authorId"].astype("int64")

    st.session_state.reviews = pd.concat(  # сохраняем изменения
        [st.session_state.reviews, df.drop("Фильм", axis=1)], ignore_index=True
    )
    # interactions: нужны authorId, movieId, rating, id, display_name, genres, genres_str, year, kp_rating,
    #               poster_preview, type_label
    df.rename(columns={'id_x': 'id', 'type': 'rating'}, inplace=True)
    df["display_name"] = movie_names
    df["year"] = movie_years
    df['rating'] = df['rating'].map({"Нейтральный": 0, "Позитивный": 1, "Негативный": 0}, na_action='ignore')

    other_data = fields_from_id(movie_ids, ["genres", "genres_str", "kp_rating", "poster_preview", "type_label"],
                                movies_indexed)
    for elem in other_data.keys():
        df[elem] = other_data[elem]

    df = df.astype({"rating": "int32", "authorId": "int32", "movieId": "int32"})
    # дообучение модели
    additional_training(df)
    st.session_state.interactions = pd.concat(  # сохраняем изменения
        [st.session_state.interactions, df.drop(["Фильм", "title", "review", "id_y", "author"], axis=1)],
        ignore_index=True
    )


def render_new_user(user_stats: pd.DataFrame) -> None:
    print("Render_recomendations")
    st.title("Создание пользователя")
    # movies_indexed = movies.set_index("id")
    movies_indexed = st.session_state.movies.set_index("id")

    if 'table_visible' not in st.session_state:  # нет table_visible в st.session_state
        st.session_state.table_visible = False
        st.session_state.data_list = [(), (), (), ()]  # фильмы, отзывы и оценки

    if not st.session_state.table_visible:  # table_visible: False
        if st.button("Добавить пользователя"):
            st.session_state.table_visible = True
            st.rerun()
    else:
        ind = get_new_index(user_stats)  # индекс нового пользователя
        st.markdown(f"### Новый пользователь, Id {ind}")

        dict_df = {
            "Фильм": st.session_state.data_list[0],
            "Заголовок": st.session_state.data_list[1],
            "Отзыв": st.session_state.data_list[2],
            "Тип": st.session_state.data_list[3]
        }
        df = pd.DataFrame(data=dict_df, dtype=str)  # для сохранения

        # список id самых обсуждаемых всех фильмов
        top_movie_pool = build_top_movie_pool(st.session_state.interactions, movies_indexed)
        movies_list = list(map(lambda movie_id: movie_label(int(movie_id), movies_indexed), top_movie_pool))
        # movies_list = tuple(map(lambda movie_id: movie_label(int(movie_id), movies_indexed),
        #                         movies_indexed.index.tolist()))  # список id всех фильмов
        with st.form("my_table_form"):
            nick_name = st.text_input("Логин пользователя")
            edited_df = get_page_table(df, movies_list)
            save_button = st.form_submit_button("Сохранить изменения")
            if save_button:  # кнопка сохранения
                if not nick_name.strip():
                    st.error("Введите имя пользователя!")
                else:
                    print(f"--- НАЧАЛЬНЫЙ DATAFRAME ---\n{edited_df}")
                    update_reviews(edited_df, nick_name, ind, movies_indexed)
                    st.success("Данные сохранены!")
                    st.session_state.data_list = [(), (), (), ()]
                    del st.session_state['table_visible']
                    st.rerun()


@st.cache_resource(show_spinner=False)
def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model("catboost_bin.json", format='json')
    return model


def main() -> None:
    if "movies" not in st.session_state:  # "id_y" == "movieId", "id_x" == id отзыва
        st.session_state.movies, st.session_state.reviews, st.session_state.interactions = load_datasets()
        # movies, reviews, interactions = load_datasets()

    if "model" not in st.session_state:
        st.session_state.model = load_model()

    # movie_stats, user_stats = build_stats(interactions, reviews)  # статистические переменные
    movie_stats, user_stats = build_stats(st.session_state.interactions,
                                          st.session_state.reviews)  # статистические переменные

    # assets = build_recommendation_assets(interactions)
    assets = build_recommendation_assets(st.session_state.interactions)

    st.sidebar.title("Навигация")
    section = st.sidebar.radio(
        "Раздел",
        ["Обзор", "Данные", "Модели", "Демо рекомендаций", "Отзывы", "Создание пользователя"],
    )
    st.sidebar.caption("Информационная система работает полностью на локальных JSON-файлах дипломного проекта.")

    movies_indexed = st.session_state.movies.set_index("id")
    if section == "Обзор":
        render_overview(user_stats)
    elif section == "Данные":
        # render_data_tab(st.session_state.movies, st.session_state.reviews, st.session_state.interactions, movie_stats,
        #                         movies_indexed)
        render_data_tab(movie_stats, movies_indexed)
    elif section == "Модели":
        render_models_tab()
    elif section == "Демо рекомендаций":  # "Демо рекомендаций"
        # render_demo_tab(st.session_state.movies, st.session_state.interactions, user_stats, assets)
        render_demo_tab(user_stats, assets)
    elif section == "Отзывы":
        # render_reviews_tab(movies, reviews)
        render_reviews_tab()
    else:
        # render_new_user(my_model, movies, reviews, interactions, user_stats)
        render_new_user(user_stats)


if __name__ == "__main__":
    main()

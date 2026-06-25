"""
app.py — BIMLearn Flask Application
=====================================
All 4 use cases are wired here.  The inference core (predictor.py)
is shared with demo_local.py and is never imported by the templates.

Routes
------
  UC-1  GET  /                          dashboard + module list
        GET  /module/<slug>             single module page (reference + actions)
  UC-2  GET  /practice/<slug>           practice page (live webcam)
        POST /api/predict               single-frame inference → JSON
  UC-3  GET  /quiz/<slug>               quiz page (timed, 5 Qs)
        POST /api/quiz/start            create QuizSession → {session_id}
        POST /api/quiz/answer           submit one answer → {correct, score, …}
        POST /api/quiz/end              finalise QuizSession
  UC-4  GET  /stats                     statistics page
        GET  /api/stats                 JSON stats (for AJAX refresh)
"""

import os
import random
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

from predictor import get_predictor, CLASS_LABELS
from models import (
    db, Word, ModuleProgress, PracticeAttempt,
    QuizSession, QuizAttempt,
    seed_words, get_stats_summary,
)

# ── App factory ────────────────────────────────────────────────────────────────
def create_app():
    app = Flask(__name__)

    # Database: SQLite for local demo, swap DATABASE_URL for PostgreSQL in prod
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:///bimlearn.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

    db.init_app(app)

    with app.app_context():
        db.create_all()
        seed_words()
        # Pre-load model so first request is fast
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.environ.get(
            "BIM_MODEL_PATH",
            os.path.join(BASE_DIR, "efficientnet_b0_bim_static_model_finetuned.keras")
        )
        get_predictor(MODEL_PATH)

    _register_routes(app)
    return app


def _register_routes(app):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.environ.get(
        "BIM_MODEL_PATH",
        os.path.join(BASE_DIR, "efficientnet_b0_bim_static_model_finetuned.keras")
    )

    # ── UC-1: Dashboard ────────────────────────────────────────────────────────
    @app.route("/")
    def dashboard():
        words = Word.query.all()
        return render_template("dashboard.html", words=words)

    # ── UC-1: Module page ──────────────────────────────────────────────────────
    @app.route("/module/<slug>")
    def module(slug):
        word = Word.query.filter_by(slug=slug).first_or_404()
        return render_template("module.html", word=word)

    # ── UC-2: Practice page ────────────────────────────────────────────────────
    @app.route("/practice/<slug>")
    def practice(slug):
        word = Word.query.filter_by(slug=slug).first_or_404()
        return render_template("practice.html", word=word)

    # ── UC-2 & UC-3: Shared inference endpoint ─────────────────────────────────
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """
        Accept one JPEG frame, return prediction JSON.
        Used by both practice and quiz pages.

        Body (multipart/form-data):
            frame   : JPEG file
            save    : "1" to persist a PracticeAttempt row (practice page only)
            word_id : int, required when save="1"
        """
        import base64

        try:
            predictor = get_predictor(MODEL_PATH)
            ct = request.content_type or ""

            if "multipart/form-data" in ct:
                jpeg_bytes = request.files["frame"].read()
            elif "application/json" in ct:
                data = request.get_json(force=True)
                b64  = data.get("frame", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                jpeg_bytes = base64.b64decode(b64)
            else:
                jpeg_bytes = request.data

            result = predictor.predict_jpeg_bytes(jpeg_bytes, smooth=False)

            # Optionally persist a practice attempt
            save    = request.form.get("save") or (request.get_json(silent=True) or {}).get("save")
            word_id = request.form.get("word_id") or (request.get_json(silent=True) or {}).get("word_id")

            if save and word_id and result["status"] == "ok":
                word_id = int(word_id)
                word    = Word.query.get(word_id)
                if word:
                    is_correct = (result["label"] == word.slug)
                    attempt = PracticeAttempt(
                        word_id    = word_id,
                        is_correct = is_correct,
                        confidence = result["confidence"],
                        predicted  = result["label"],
                    )
                    db.session.add(attempt)
                    word.progress.update_from_attempt(is_correct)
                    db.session.commit()

            return jsonify(result)

        except Exception as exc:
            app.logger.exception("Prediction error")
            return jsonify({"status": "error", "message": str(exc)}), 500

    # ── UC-3: Quiz page ────────────────────────────────────────────────────────
    @app.route("/quiz/<slug>")
    def quiz(slug):
        if slug == "general":
            is_general = True
            # Mock structural object container parameters to handle the template rules smoothly
            class DummyWord:
                slug = "general"
                label_ms = "Umum"
                emoji = "🧩"
            word = DummyWord()
        else:
            is_general = False
            word = Word.query.filter_by(slug=slug).first_or_404()
            
        return render_template("quiz.html", word=word, is_general=is_general, all_slugs=CLASS_LABELS)

    @app.route("/api/quiz/start", methods=["POST"])
    def quiz_start():
        """Create a new QuizSession and return its id + first question word."""
        data        = request.get_json(force=True)
        anchor_slug = data.get("anchor","general")  # word the quiz was launched from
        
        if anchor_slug == "general":
            n_questions = int(data.get("n_questions", 5))
            # Build randomised combinations from global list array items
            pool = [random.choice(CLASS_LABELS) for _ in range(n_questions)]
        else:
            n_questions = 3
            pool = [anchor_slug for _ in range(n_questions)]

        session = QuizSession()
        session.total = 0
        session.score = 0
        db.session.add(session)
        db.session.commit()

        return jsonify({
            "session_id": session.id,
            "questions":  pool,          # ordered list of word slugs
            "total":      n_questions,
        })

    @app.route("/api/quiz/answer", methods=["POST"])
    def quiz_answer():
        """
        Submit a single quiz answer.
        Body JSON:
            session_id  : int
            word_slug   : str   target word for this question
            predicted   : str   what the model returned (or "timeout")
            confidence  : float
            timed_out   : bool
            time_ms     : int   milliseconds taken
        """
        data       = request.get_json(force=True)
        session    = QuizSession.query.get_or_404(data["session_id"])
        word       = Word.query.filter_by(slug=data["word_slug"]).first_or_404()
        predicted  = data.get("predicted", "")
        timed_out  = bool(data.get("timed_out", False))
        is_correct = (not timed_out) and (predicted == word.slug)
        confidence = float(data.get("confidence", 0.0))
        time_ms    = int(data.get("time_ms", 0))

        attempt = QuizAttempt(
            session_id   = session.id,
            word_id      = word.id,
            is_correct   = is_correct,
            confidence   = confidence,
            predicted    = predicted,
            timed_out    = timed_out,
            time_taken_ms= time_ms,
        )
        db.session.add(attempt)

        session.total += 1
        if is_correct:
            session.score += 1

        # Update word progress
        word.progress.update_from_attempt(is_correct)
        db.session.commit()

        return jsonify({
            "is_correct":   is_correct,
            "score":        session.score,
            "total":        session.total,
            "correct_slug": word.slug,
            "correct_label": word.label_ms,
        })

    @app.route("/api/quiz/end", methods=["POST"])
    def quiz_end():
        data    = request.get_json(force=True)
        session = QuizSession.query.get_or_404(data["session_id"])
        session.finished_at = datetime.utcnow()
        db.session.commit()
        return jsonify({
            "score":    session.score,
            "total":    session.total,
            "accuracy": session.accuracy,
        })

    # ── UC-4: Statistics page ──────────────────────────────────────────────────
    @app.route("/stats")
    def stats():
        summary = get_stats_summary()
        return render_template("stats.html", **summary)

    @app.route("/api/stats")
    def api_stats():
        return jsonify(get_stats_summary())

    # ── Utility ────────────────────────────────────────────────────────────────
    @app.route("/api/words")
    def api_words():
        words = Word.query.all()
        return jsonify([{
            "id": w.id, "slug": w.slug,
            "label_ms": w.label_ms, "label_en": w.label_en,
            "emoji": w.emoji,
            "progress": w.progress.progress_pct if w.progress else 0,
            "accuracy": w.progress.accuracy     if w.progress else 0,
        } for w in words])


# ── Entry point ────────────────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

with app.app_context():
    #db.create_all()  # This ensures Postgres tables are generated if they don't exist
    #seed_words()     # Populates your initial vocabulary words list
    pass
    

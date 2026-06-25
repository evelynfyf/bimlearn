"""
models.py — SQLAlchemy database models for BIMLearn
=====================================================
Schema covers all 4 use cases:
  UC-1  Mengurus Modul Pembelajaran  →  Word, ModuleProgress
  UC-2  Berlatih Isyarat             →  PracticeAttempt
  UC-3  Mengambil Kuiz               →  QuizSession, QuizAttempt
  UC-4  Lihat Statistik Prestasi     →  computed from all tables above

Database
--------
Local/demo : SQLite  → file `bimlearn.db`  (zero setup)
Production : PostgreSQL → set DATABASE_URL env var

Switching databases requires only one env var change; SQLAlchemy handles the rest.
"""

import os
from datetime import datetime, date, timedelta

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# ── Word ───────────────────────────────────────────────────────────────────────
class Word(db.Model):
    """
    One row per BIM word supported by the classifier.
    Seeded once at app startup via seed_words().
    """
    __tablename__ = "words"

    id          = db.Column(db.Integer, primary_key=True)
    slug        = db.Column(db.String(32), unique=True, nullable=False)  # "makan"
    label_ms    = db.Column(db.String(64), nullable=False)               # "Makan"
    label_en    = db.Column(db.String(64), nullable=False)               # "Eat"
    emoji       = db.Column(db.String(8),  nullable=False)
    description = db.Column(db.Text, default="")

    # Relationships
    progress    = db.relationship("ModuleProgress",  back_populates="word", uselist=False)
    practices   = db.relationship("PracticeAttempt", back_populates="word")
    quiz_items  = db.relationship("QuizAttempt",     back_populates="target_word")

    def __repr__(self):
        return f"<Word {self.slug}>"


# ── ModuleProgress ─────────────────────────────────────────────────────────────
class ModuleProgress(db.Model):
    """
    Tracks how well the user knows each word.
    Progress (0–100) is updated after each practice/quiz attempt.
    """
    __tablename__ = "module_progress"

    id           = db.Column(db.Integer, primary_key=True)
    word_id      = db.Column(db.Integer, db.ForeignKey("words.id"), unique=True, nullable=False)
    progress_pct = db.Column(db.Float,   default=0.0)      # 0–100
    attempts     = db.Column(db.Integer, default=0)
    correct      = db.Column(db.Integer, default=0)
    last_seen    = db.Column(db.DateTime, nullable=True)

    word         = db.relationship("Word", back_populates="progress")

    @property
    def accuracy(self):
        """Return accuracy as a percentage, or 0 if never attempted."""
        return round((self.correct / self.attempts) * 100) if self.attempts else 0

    def update_from_attempt(self, is_correct: bool):
        self.attempts  += 1
        self.correct   += int(is_correct)
        self.last_seen  = datetime.utcnow()
        # Simple EMA-style progress: weight recent performance heavily
        new_pct = (self.correct / self.attempts) * 100
        self.progress_pct = round(new_pct, 1)


# ── PracticeAttempt ────────────────────────────────────────────────────────────
class PracticeAttempt(db.Model):
    """
    One row per individual sign attempt during a free-practice session (UC-2).
    """
    __tablename__ = "practice_attempts"

    id           = db.Column(db.Integer, primary_key=True)
    word_id      = db.Column(db.Integer, db.ForeignKey("words.id"), nullable=False)
    is_correct   = db.Column(db.Boolean, nullable=False)
    confidence   = db.Column(db.Float,   nullable=False)   # model confidence 0–1
    predicted    = db.Column(db.String(32), nullable=False) # what the model said
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    word         = db.relationship("Word", back_populates="practices")


# ── QuizSession ────────────────────────────────────────────────────────────────
class QuizSession(db.Model):
    """
    One row per quiz run (UC-3). Contains aggregate results.
    Individual question results are in QuizAttempt.
    """
    __tablename__ = "quiz_sessions"

    id           = db.Column(db.Integer, primary_key=True)
    score        = db.Column(db.Integer, default=0)         # correct answers
    total        = db.Column(db.Integer, default=0)         # questions asked
    started_at   = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at  = db.Column(db.DateTime, nullable=True)

    attempts     = db.relationship("QuizAttempt", back_populates="session",
                                   cascade="all, delete-orphan")

    @property
    def accuracy(self):
        return round((self.score / self.total) * 100) if self.total else 0

    @property
    def duration_seconds(self):
        if self.finished_at:
            return (self.finished_at - self.started_at).seconds
        return None


# ── QuizAttempt ────────────────────────────────────────────────────────────────
class QuizAttempt(db.Model):
    """
    One row per question within a quiz session.
    """
    __tablename__ = "quiz_attempts"

    id             = db.Column(db.Integer, primary_key=True)
    session_id     = db.Column(db.Integer, db.ForeignKey("quiz_sessions.id"), nullable=False)
    word_id        = db.Column(db.Integer, db.ForeignKey("words.id"),         nullable=False)
    is_correct     = db.Column(db.Boolean, nullable=False)
    confidence     = db.Column(db.Float,   default=0.0)
    predicted      = db.Column(db.String(32), nullable=True)
    timed_out      = db.Column(db.Boolean, default=False)
    time_taken_ms  = db.Column(db.Integer, nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    session        = db.relationship("QuizSession",  back_populates="attempts")
    target_word    = db.relationship("Word",         back_populates="quiz_items")


# ── Seed data ──────────────────────────────────────────────────────────────────
WORD_SEED = [
    ("air",    "Air",    "Water",  "💧", "Isyarat untuk air minuman."),
    ("demam",  "Demam",  "Fever",  "🤒", "Isyarat apabila seseorang sakit demam."),
    ("dengar", "Dengar", "Listen", "👂", "Isyarat untuk mendengar"),
    ("makan",  "Makan",  "Eat",    "🍽️", "Isyarat untuk makan."),
    ("minum",  "Minum",  "Drink",  "🥛", "Isyarat untuk minum."),
    ("salah",  "Salah",  "Wrong",  "❌", "Isyarat untuk menyatakan sesuatu itu salah."),
    ("saya",   "Saya",   "I / Me", "👤", "Isyarat merujuk kepada diri sendiri."),
    ("senyap", "Senyap", "Quiet",  "🤫", "Isyarat meminta seseorang berdiam diri."),
    ("tidur",  "Tidur",  "Sleep",  "😴", "Isyarat untuk tidur atau berehat."),
    ("waktu",  "Waktu",  "Time",   "⏰", "Isyarat untuk menyatakan masa atau waktu."),
]

def seed_words():
    """Insert Word and ModuleProgress rows if they don't exist yet."""
    for slug, ms, en, emoji, desc in WORD_SEED:
        if not Word.query.filter_by(slug=slug).first():
            word = Word(slug=slug, label_ms=ms, label_en=en,
                        emoji=emoji, description=desc)
            db.session.add(word)
            db.session.flush()   # get word.id before commit
            db.session.add(ModuleProgress(word_id=word.id))
    db.session.commit()


# ── Statistics helpers (UC-4) ──────────────────────────────────────────────────
def get_stats_summary():
    """
    Compute the summary statistics shown on the Statistics screen.
    Returns a plain dict — safe to pass to Jinja2 templates or jsonify().
    """
    from sqlalchemy import func

    total_practice = PracticeAttempt.query.count()
    total_quiz_sessions = QuizSession.query.filter(
        QuizSession.finished_at.isnot(None)
    ).count()
    total_sessions = total_practice + total_quiz_sessions  # combined activity count

    # Overall accuracy across both practice and quiz
    practice_correct = PracticeAttempt.query.filter_by(is_correct=True).count()
    quiz_correct = db.session.query(func.sum(QuizSession.score)).scalar() or 0
    quiz_total   = db.session.query(func.sum(QuizSession.total)).scalar() or 0
    total_correct = practice_correct + quiz_correct
    total_attempts = total_practice + (quiz_total or 0)
    overall_accuracy = round((total_correct / total_attempts) * 100) if total_attempts else 0

    # Words mastered = accuracy >= 70% with at least 3 attempts
    mastered_count = ModuleProgress.query.filter(
        ModuleProgress.attempts >= 3,
        ModuleProgress.progress_pct >= 70
    ).count()

    # Streak: count consecutive days with at least one activity
    streak = _compute_streak()

    # Per-word accuracy for the stats table and weak words
    words_data = []
    for word in Word.query.all():
        prog = word.progress
        words_data.append({
            "slug":       word.slug,
            "label":      word.label_ms,
            "emoji":      word.emoji,
            "accuracy":   prog.accuracy if prog else 0,
            "attempts":   prog.attempts if prog else 0,
            "progress":   prog.progress_pct if prog else 0,
        })

    weak_words = sorted(
        [w for w in words_data if w["attempts"] > 0],
        key=lambda x: x["accuracy"]
    )[:3]

    # Daily accuracy for the trend chart (last 14 days)
    daily = _daily_accuracy(days=14)

    return {
        "total_sessions":     total_sessions,
        "overall_accuracy":   overall_accuracy,
        "words_mastered":     f"{mastered_count}/10",
        "streak":             f"{streak} hari",
        "words_data":         words_data,
        "weak_words":         weak_words,
        "daily_chart":        daily,
    }


def _compute_streak():
    """Count how many consecutive calendar days had at least one practice/quiz attempt."""
    from sqlalchemy import func
    
    # Function to normalise whatever the database returns into standard date object (local SQLite or PostgreSQL)
    def to_date_obj(val):
        if isinstance(val, datetime):
            return val.date()
        elif isinstance(val, date):
            return val
        elif isinstance(val, str) and val:
            return datetime.strptime(val.split()[0], "%Y-%m-%d").date()
        else:
            raise ValueError(f"Unexpected date value: {val}")
    
    practice_days = {
        to_date_obj(r[0])
        for r in db.session.query(func.date(PracticeAttempt.created_at)).distinct().all()
        if r[0]
    } - {None}  # Remove None values if any

    quiz_days = {
        to_date_obj(r[0])
        for r in db.session.query(func.date(QuizSession.started_at))
        .filter(QuizSession.finished_at.isnot(None)).distinct().all()
        if r[0]
    } - {None}  # Remove None values if any

    active_days = practice_days | quiz_days
    streak = 0
    check_date = date.today()
    while check_date in active_days:
        streak += 1
        check_date -= timedelta(days=1)
    return streak


def _daily_accuracy(days=14):
    """Return a list of {date, accuracy} dicts for the last N days."""
    from sqlalchemy import cast, Date as SADate, func

    result = []
    for i in range(days - 1, -1, -1):
        d = date.today() - timedelta(days=i)

        # Cast created_at to an actual Date object so PostgreSQL matches d correctly
        # (SQLite does this automatically, but Postgres needs explicit casting)

        # Practice accuracy for this day
        p_total = PracticeAttempt.query.filter(
            cast(PracticeAttempt.created_at, SADate) == d
        ).count()
        p_correct = PracticeAttempt.query.filter(
            cast(PracticeAttempt.created_at, SADate) == d,
            PracticeAttempt.is_correct == True
        ).count()

        q_total = QuizAttempt.query.join(QuizSession).filter(
            cast(QuizSession.started_at, SADate) == d,
            QuizSession.finished_at.isnot(None)
        ).count()
        q_correct = QuizAttempt.query.join(QuizSession).filter(
            cast(QuizSession.started_at, SADate) == d,
            QuizSession.finished_at.isnot(None),
            QuizAttempt.is_correct == True
        ).count()

        total = p_total + q_total
        correct = p_correct + q_correct
        acc = round((correct / total) * 100) if total > 0 else None
        
        result.append({
            "date": d.strftime("%d/%m"), 
            "accuracy": acc
        })
    return result

"""Orchestrates a verification attempt end to end.

Sequence: detect+align -> liveness -> identity -> decide -> persist.

The orchestrator gathers evidence; it does not decide. The verdict comes from
`core.decision.decide`, a pure function. Keeping evidence-gathering and
decision-making separate is what allows the security-critical rule set to be tested
exhaustively without mocking models, HTTP or a database.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from backend.app.core.decision import (
    Decision,
    Reason,
    Thresholds,
    VerificationInput,
    decide,
)
from backend.app.models.db import Attendance, FaceEmbedding, Session as ClassSession, User
from backend.app.preprocessing_bridge import normalize
from backend.app.services.recognition import best_match, deserialize


class VerificationService:
    def __init__(self, face_processor, liveness, recognition, thresholds: Thresholds,
                 identity_margin: float = 0.0):
        self.faces = face_processor
        self.liveness = liveness
        self.recognition = recognition
        self.thresholds = thresholds
        self.identity_margin = identity_margin

    # ---------- evidence gathering ----------

    def _load_templates(self, db: OrmSession) -> tuple[dict[str, np.ndarray], dict[str, User]]:
        """Load active users' templates for the CURRENT recognition model version.

        Filtering on model_version is essential: embeddings from a different backbone
        are not comparable, and silently mixing them produces similarity scores that
        look plausible and mean nothing.
        """
        rows = db.execute(
            select(FaceEmbedding, User)
            .join(User, User.id == FaceEmbedding.user_id)
            .where(FaceEmbedding.model_version == self.recognition.model_version)
            .where(User.status == "active")
        ).all()

        templates, users = {}, {}
        for emb, user in rows:
            templates[user.id] = deserialize(emb.embedding, emb.dim)
            users[user.id] = user
        return templates, users

    def _open_session(self, db: OrmSession, session_id: str | None) -> ClassSession | None:
        now = dt.datetime.now(dt.timezone.utc)
        stmt = select(ClassSession).where(
            ClassSession.start_time <= now, ClassSession.end_time >= now)
        if session_id:
            stmt = stmt.where(ClassSession.id == session_id)
        return db.execute(stmt).scalars().first()

    # ---------- the attempt ----------

    def verify(self, db: OrmSession, frames_bgr: list[np.ndarray],
               session_id: str | None = None, mark_attendance: bool = True) -> dict:
        n_faces, liveness_score, identity_score, matched_id = 0, None, None, None
        user, klass = None, None

        try:
            # 1. detect and align every frame with the same code path used in training
            crops, per_frame_counts = [], []
            for frame in frames_bgr:
                found = self.faces.detect_all(frame)
                per_frame_counts.append(len(found))
                if len(found) == 1:
                    crops.append(normalize(found[0].crop))

            # A frame showing two faces anywhere in the burst is treated as multiple
            # faces overall: an attacker must not be able to slip a second person into
            # part of the capture window.
            n_faces = max(per_frame_counts) if per_frame_counts else 0
            if n_faces == 1 and not crops:
                n_faces = 0

            if n_faces == 1:
                liveness_score = self.liveness.score(crops)

                # Identity is computed only if liveness passed, so an attack
                # presentation is never matched against the registry.
                if liveness_score >= self.thresholds.liveness:
                    templates, users = self._load_templates(db)
                    probe, n = self.recognition.embed(frames_bgr[len(frames_bgr) // 2])
                    if probe is not None and templates:
                        m = best_match(probe, templates, self.thresholds.identity,
                                       self.identity_margin)
                        identity_score = m.similarity
                        matched_id = m.user_id
                        user = users.get(m.user_id) if m.user_id else None
                    else:
                        identity_score = 0.0

            klass = self._open_session(db, session_id)
            already = False
            if user and klass:
                already = db.execute(
                    select(Attendance).where(Attendance.user_id == user.id,
                                             Attendance.session_id == klass.id)
                ).scalars().first() is not None

            decision = decide(
                VerificationInput(
                    n_faces=n_faces,
                    liveness_score=liveness_score,
                    identity_score=identity_score,
                    matched_user_id=matched_id,
                    user_is_active=(user.status == "active") if user else True,
                    session_is_open=klass is not None,
                    already_marked=already,
                ),
                self.thresholds,
            )
        except Exception:
            # Any unexpected failure rejects. Never fall through to an approval.
            decision = Decision(approved=False, reason=Reason.SYSTEM_ERROR)

        payload = decision.to_dict()

        if decision.approved and mark_attendance and klass is not None:
            row = Attendance(user_id=decision.user_id, session_id=klass.id,
                             identity_confidence=decision.identity_confidence or 0.0,
                             liveness_confidence=decision.liveness_confidence or 0.0)
            db.add(row)
            try:
                db.commit()
                payload["attendance_id"] = row.id
            except IntegrityError:
                # Lost a race against a concurrent request for the same (user, session).
                # The database constraint is what actually prevents the duplicate; this
                # turns the violation into the correct user-facing answer.
                db.rollback()
                payload = {"decision": "rejected", "reason": Reason.ALREADY_MARKED.value}

        self._audit(db, payload, decision, n_faces, klass)
        return payload

    def _audit(self, db, payload, decision, n_faces, klass) -> None:
        """Record the outcome. No images, no embeddings — see models/db.py."""
        from backend.app.models.db import VerificationAttempt
        db.add(VerificationAttempt(
            session_id=klass.id if klass else None,
            matched_user_id=decision.user_id if decision.approved else None,
            approved=decision.approved,
            reason=payload.get("reason"),
            liveness_confidence=decision.liveness_confidence,
            identity_confidence=decision.identity_confidence if decision.approved else None,
            n_faces=n_faces,
            liveness_model_version=getattr(self.liveness, "model_version", "unknown"),
        ))
        db.commit()

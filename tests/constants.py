"""Shared test constants for the kindle test suite."""

from __future__ import annotations

SAMPLE_FEATURE_SPEC: dict = {
    "app_name": "TaskFlow",
    "idea": "a task management app",
    "core_features": ["task CRUD", "auth"],
    "tech_constraints": ["React frontend"],
}

SAMPLE_QUESTIONS: list[dict] = [
    {
        "question": "What are the must-have features for the MVP?",
        "category": "core_functionality",
        "recommended_answer": "User login, dashboard, and CRUD for tasks",
    },
    {
        "question": "Is authentication required?",
        "category": "user_model",
        "recommended_answer": "Yes, email/password auth",
    },
    {
        "question": "What is the target platform?",
        "category": "platform",
        "recommended_answer": "Web application (SPA)",
    },
]

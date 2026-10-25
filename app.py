from flask import Flask, render_template, request, jsonify

from src.predict import predict_failure
from src.config import METRICS_PATH

import json
import os
import re
import time
import resend

from dotenv import load_dotenv

# Loads variables from a local .env file if present (for local dev).
# On Render, environment variables are set directly in the dashboard,
# so there's no .env file there -- this call is a harmless no-op in
# that case since it just won't find anything to load.
load_dotenv()


app = Flask(__name__)


# ---------------- Feedback form config ----------------
#
# Resend API key is read from an environment variable, never hardcoded --
# set RESEND_API_KEY locally (e.g. in a .env file, not committed to git)
# and as an environment variable on Render. FEEDBACK_TO_EMAIL is where
# submissions land; FEEDBACK_FROM_EMAIL must be a domain you've verified
# with Resend (their sandbox "onboarding@resend.dev" sender works without
# verification for testing, but has tighter sending limits).

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
FEEDBACK_TO_EMAIL = os.environ.get("FEEDBACK_TO_EMAIL", "athar@example.com")
FEEDBACK_FROM_EMAIL = os.environ.get("FEEDBACK_FROM_EMAIL", "onboarding@resend.dev")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Simple in-memory rate limit: max 5 feedback submissions per IP per hour.
# This is intentionally lightweight (resets on server restart, doesn't
# scale across multiple server instances) -- it's meant to deter casual
# abuse of a public form, not to be a robust production rate limiter.
_feedback_submissions = {}
FEEDBACK_RATE_LIMIT = 5
FEEDBACK_RATE_WINDOW_SECONDS = 3600


def is_rate_limited(ip):
    now = time.time()
    timestamps = _feedback_submissions.get(ip, [])
    timestamps = [t for t in timestamps if now - t < FEEDBACK_RATE_WINDOW_SECONDS]
    _feedback_submissions[ip] = timestamps
    return len(timestamps) >= FEEDBACK_RATE_LIMIT


def record_submission(ip):
    _feedback_submissions.setdefault(ip, []).append(time.time())


# Load model metrics + PR curve once at startup (not per-request -- this
# file is small and only changes when the model is retrained, so there's
# no reason to re-read it from disk on every page view).
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        # Lets the app still boot if metrics.json hasn't been generated
        # yet (e.g. fresh clone before running the training script), with
        # an obvious "not available" state in the UI rather than crashing.
        return None


METRICS = load_metrics()


@app.route("/")
def home():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if len(text) > 2000:
        return jsonify({
            "error": "Text is too long (max 2000 characters)."
        }), 400

    result = predict_failure(text)

    return jsonify(result)


@app.route("/api/metrics")
def metrics():
    if METRICS is None:
        return jsonify({"error": "Metrics not available."}), 404

    return jsonify(METRICS)


@app.route("/api/feedback", methods=["POST"])
def feedback():
    if not RESEND_API_KEY:
        # Fails clearly rather than pretending to succeed -- if someone
        # clones this repo without setting up Resend, they should see an
        # honest error, not a silently-dropped form submission.
        return jsonify({
            "error": "Feedback isn't configured on this server yet."
        }), 503

    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip()
    message = (data.get("message") or "").strip()

    if not email or not EMAIL_PATTERN.match(email):
        return jsonify({"error": "Please enter a valid email address."}), 400

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > 5000:
        return jsonify({
            "error": "Message is too long (max 5000 characters)."
        }), 400

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if is_rate_limited(client_ip):
        return jsonify({
            "error": "Too many submissions. Please try again later."
        }), 429

    # Recorded before attempting the send, not after -- a flood of
    # invalid/failing submissions should still be throttled. Counting
    # only successes would let someone retry indefinitely as long as
    # each attempt errors out.
    record_submission(client_ip)

    try:
        resend.Emails.send({
            "from": FEEDBACK_FROM_EMAIL,
            "to": FEEDBACK_TO_EMAIL,
            "reply_to": email,
            "subject": "NeuroGuard ML -- new feedback submission",
            "text": f"From: {email}\n\n{message}"
        })
    except Exception:
        # Don't leak internal error details (API key issues, network
        # errors, etc.) to the client -- log server-side in a real
        # deployment, return a generic message here.
        return jsonify({
            "error": "Could not send feedback right now. Please try again later."
        }), 502

    return jsonify({"success": True})


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(

        host="0.0.0.0",

        port=port,

        debug=debug_mode

    )
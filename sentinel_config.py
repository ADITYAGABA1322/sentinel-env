from __future__ import annotations

import os


# Stakes >= this value means the adversarial specialist starts poisoning.
ADVERSARIAL_TRIGGER_STAKES = 0.70

# Stakes >= this value means the grader should reward verification/evasion.
ADVERSARIAL_AWARENESS_STAKES = ADVERSARIAL_TRIGGER_STAKES

# Stakes >= this value means unchecked adversarial poison ends the episode.
# This is intentionally higher than ADVERSARIAL_TRIGGER_STAKES: lower-stakes poison
# damages reward and trust, while peak-stakes poison represents mission failure.
CRITICAL_POISON_STAKES = 0.85

# Verifying means "ask specialist, then cross-check". Slow specialists still carry
# their own step cost, so verify(AccurateSlow) costs 2 + 1 = 3 steps.
VERIFY_EXTRA_STEP_COST = 1

# In-memory session store limits. This deployment is intentionally single-worker;
# use Redis/sticky sessions before increasing workers.
SESSION_TTL_SECONDS = int(os.environ.get("SENTINEL_SESSION_TTL_SECONDS", "1800"))
SESSION_MAX_ACTIVE = int(os.environ.get("SENTINEL_SESSION_MAX_ACTIVE", "256"))
SESSION_BACKEND = "single_process_memory"


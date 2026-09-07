"""Live end-to-end demo against the deployed cluster."""
import base64, json, subprocess, sys, time
import cv2, numpy as np, requests

API = "http://localhost:18000"
KEY = subprocess.run(["kubectl","-n","attendance","get","secret","attendance-secrets",
                      "-o","jsonpath={.data.API_KEY}"], capture_output=True, text=True).stdout
KEY = base64.b64decode(KEY).decode()
H = {"X-API-Key": KEY}

def hdr(t): print(f"\n\033[1m{'='*66}\n{t}\n{'='*66}\033[0m")
def ok(t):  print(f"  \033[32m✓\033[0m {t}")
def bad(t): print(f"  \033[31m✗\033[0m {t}")

def burst(n=8, w=640, h=480):
    return [("frames", (f"{i}.jpg", cv2.imencode(".jpg",
            (np.random.rand(h,w,3)*255).astype(np.uint8))[1].tobytes(), "image/jpeg"))
            for i in range(n)]

hdr("1. SERVICE HEALTH")
r = requests.get(f"{API}/health").json()
print(json.dumps(r, indent=2))
ok("liveness model loaded") if r["liveness_model_loaded"] else bad("liveness NOT loaded")
ok("recognition model loaded") if r["recognition_model_loaded"] else bad("recognition NOT loaded")
rd = requests.get(f"{API}/ready")
ok(f"/ready {rd.status_code} {rd.json()}")

hdr("2. AUTHENTICATION IS ENFORCED")
for label, hh in [("no key", {}), ("wrong key", {"X-API-Key":"nope"}), ("valid key", H)]:
    c = requests.get(f"{API}/users", headers=hh).status_code
    (ok if (c==200)==(label=="valid key") else bad)(f"{label:<12} -> HTTP {c}")
c = requests.get(f"{API}/health").status_code
ok(f"probes stay public -> /health HTTP {c}")

hdr("3. REGISTER A USER AND OPEN A SESSION")
import datetime as dt
now = dt.datetime.now(dt.timezone.utc)
sid = f"UPES{int(time.time())%100000}"
u = requests.post(f"{API}/users", headers=H, json={
    "student_id": sid, "name": "Madhav Sharma", "email": f"{sid}@upes.ac.in"}).json()
ok(f"user {u['name']} ({u['student_id']}) id={u['id'][:8]}...")
s = requests.post(f"{API}/sessions", headers=H, json={
    "name": "Demo Lecture",
    "start_time": (now - dt.timedelta(minutes=5)).isoformat(),
    "end_time": (now + dt.timedelta(hours=3)).isoformat()}).json()
ok(f"session '{s['name']}' open={s['is_open']} id={s['id'][:8]}...")

hdr("4. DUPLICATE REGISTRATION IS REJECTED")
c = requests.post(f"{API}/users", headers=H, json={
    "student_id": sid, "name": "Impostor", "email": f"{sid}@upes.ac.in"}).status_code
(ok if c==409 else bad)(f"same student_id again -> HTTP {c} (409 expected)")

hdr("5. ENROLLMENT REFUSES UNUSABLE IMAGES")
imgs = [("images", (f"{i}.jpg", cv2.imencode(".jpg",
        (np.random.rand(480,640,3)*255).astype(np.uint8))[1].tobytes(), "image/jpeg"))
        for i in range(4)]
r = requests.post(f"{API}/enrollment", headers=H, data={"user_id": u["id"]}, files=imgs)
(ok if r.status_code==422 else bad)(f"4 images with no face -> HTTP {r.status_code}")
print(f"     {r.json().get('detail','')}")

hdr("6. VERIFICATION — NO FACE PRESENT")
t0=time.perf_counter()
r = requests.post(f"{API}/verify", headers=H, files=burst(), data={"session_id": s["id"]})
dt_ms=(time.perf_counter()-t0)*1000
d = r.json()
(ok if d["reason"]=="NO_FACE" else bad)(f"decision={d['decision']} reason={d['reason']}  ({dt_ms:.0f} ms)")
ok("rejection leaks no identity") if d.get("user_id") is None else bad("identity leaked!")

hdr("7. NO ATTENDANCE WAS MARKED")
att = requests.get(f"{API}/attendance", headers=H).json()
(ok if len(att)==0 else bad)(f"attendance rows: {len(att)} (0 expected — nothing verified)")

hdr("8. AUDIT TRAIL (no images, no embeddings)")
q = subprocess.run(["kubectl","-n","attendance","exec","postgres-0","--",
    "psql","-U","attendance","-d","attendance","-t","-c",
    "SELECT reason, n_faces, approved, liveness_model_version, created_at "
    "FROM verification_attempts ORDER BY created_at DESC LIMIT 3;"],
    capture_output=True, text=True).stdout.strip()
print(q if q else "  (none)")
cols = subprocess.run(["kubectl","-n","attendance","exec","postgres-0","--",
    "psql","-U","attendance","-d","attendance","-t","-c",
    "SELECT string_agg(column_name, ', ') FROM information_schema.columns "
    "WHERE table_name='verification_attempts';"], capture_output=True, text=True).stdout.strip()
print(f"\n  columns: {cols}")
forbidden = {"image","embedding","frame","photo"}
leaked = [c for c in cols.replace(" ","").split(",") if any(f in c for f in forbidden)]
(ok if not leaked else bad)(f"no biometric columns in the audit log ({leaked or 'none'})")

hdr("9. RATE LIMITING ON /verify")
codes=[]
for i in range(34):
    codes.append(requests.post(f"{API}/verify", headers=H, files=burst(1)).status_code)
n429 = codes.count(429)
(ok if n429>0 else bad)(f"34 rapid attempts -> {codes.count(200)} accepted, {n429} rate-limited (429)")

hdr("10. DECISION ENGINE — EVERY REJECTION PATH")
sys.path.insert(0,".")
from backend.app.core.decision import decide, VerificationInput, Thresholds
T = Thresholds(liveness=0.4535, identity=0.5)
base = dict(n_faces=1, liveness_score=0.95, identity_score=0.9, matched_user_id="u1",
            user_is_active=True, session_is_open=True, already_marked=False)
cases = [
    ("all checks pass",            {}),
    ("no face",                    {"n_faces":0}),
    ("two people in frame",        {"n_faces":2}),
    ("photo/screen attack",        {"liveness_score":0.05}),
    ("liveness inconclusive",      {"liveness_score":0.40}),
    ("unknown person",             {"matched_user_id":None}),
    ("identity uncertain",         {"identity_score":0.30}),
    ("session closed",             {"session_is_open":False}),
    ("already marked",             {"already_marked":True}),
    ("inference failed",           {"liveness_score":None}),
]
for label, over in cases:
    kw = {**base, **over}
    d = decide(VerificationInput(**kw), T)
    tag = "\033[32mAPPROVED\033[0m" if d.approved else f"\033[33m{d.reason.value}\033[0m"
    print(f"  {label:<24} -> {tag}")

print("\n\033[1mDemo complete.\033[0m")

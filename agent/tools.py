import re, datetime
from typing import Optional

_captured_leads = []

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    lead = {
        "lead_id": f"LEAD-{len(_captured_leads) + 1001}",
        "name": name, "email": email, "platform": platform,
        "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "new", "source": "inflx-autostream-agent",
    }
    _captured_leads.append(lead)
    print("\n" + "=" * 60)
    print("LEAD CAPTURED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Lead captured successfully: {name}, {email}, {platform}")
    print(f"  Lead ID  : {lead['lead_id']}")
    print(f"  Timestamp: {lead['captured_at']}")
    print("=" * 60 + "\n")
    return {"success": True, "lead_id": lead["lead_id"]}

def validate_email(email: str) -> bool:
    return bool(re.match(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$", email.strip()))

def extract_lead_fields_from_text(text: str, existing: dict) -> dict:
    result = dict(existing)
    if not result.get("email"):
        m = re.search(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", text)
        if m:
            result["email"] = m.group(0)
    platforms = ["youtube","instagram","tiktok","facebook","twitter","linkedin","twitch","snapchat"]
    if not result.get("platform"):
        for p in platforms:
            if p in text.lower():
                result["platform"] = p.capitalize()
                break
    return result

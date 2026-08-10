"""Google integrations — Gmail, Calendar, Sheets, Drive (OAuth-based)."""
from __future__ import annotations

import base64
import logging
from typing import Any

from app.config import get_settings
from app.models import Integration, User

logger = logging.getLogger(__name__)
settings = get_settings()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

PROVIDER_SCOPES = {
    "gmail": ["https://www.googleapis.com/auth/gmail.readonly"],
    "google_calendar": ["https://www.googleapis.com/auth/calendar"],
    "google_sheets": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
    "google_drive": ["https://www.googleapis.com/auth/drive.readonly"],
}

PROVIDER_LABELS = {
    "gmail": "📧 Gmail",
    "google_calendar": "📅 Google Calendar",
    "google_sheets": "📊 Google Sheets",
    "google_drive": "🗂 Google Drive",
    "google_docs": "📄 Google Docs",
}


def is_configured() -> bool:
    """True when OAuth credentials are present.

    Reads fresh settings so env changes are picked up without a restart.
    """
    fresh = get_settings()
    return bool(fresh.google_client_id and fresh.google_client_secret)


def build_auth_url(state: str, provider: str = "gmail") -> str:
    """Build the Google OAuth consent URL for a provider."""
    from urllib.parse import urlencode

    scopes = " ".join(PROVIDER_SCOPES.get(provider, PROVIDER_SCOPES["gmail"]))
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "access_type": "offline",
        "prompt": "consent",
        "state": f"{provider}:{state}",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


async def exchange_code(code: str) -> dict[str, Any]:
    """Exchange OAuth code for tokens."""
    import httpx

    data = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_redirect_uri,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post("https://oauth2.googleapis.com/token", data=data)
        resp.raise_for_status()
        return resp.json()


async def save_integration(user: User, provider: str, tokens: dict[str, Any]) -> Integration:
    """Store integration tokens for a user."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        existing = (
            db.query(Integration)
            .filter(Integration.user_id == user.id, Integration.provider == provider)
            .first()
        )
        if existing:
            existing.credentials = tokens
            db.commit()
            db.refresh(existing)
            return existing

        integration = Integration(user_id=user.id, provider=provider, credentials=tokens)
        db.add(integration)
        db.commit()
        db.refresh(integration)
        return integration
    finally:
        db.close()


def _get_tokens(integration: Integration) -> dict[str, Any]:
    return integration.credentials or {}


async def _refresh_access_token(integration: Integration) -> str | None:
    """Refresh an expired access token if a refresh token is available."""
    creds = _get_tokens(integration)
    refresh_token = creds.get("refresh_token")
    if not refresh_token:
        return None
    import httpx

    data = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://oauth2.googleapis.com/token", data=data)
            resp.raise_for_status()
            new_tokens = resp.json()
        access_token = new_tokens.get("access_token")
        if access_token:
            existed = _get_tokens(integration) or {}
            existed["access_token"] = access_token
            existed["refresh_token"] = refresh_token
            from app.database import SessionLocal

            db = SessionLocal()
            try:
                integration.credentials = existed
                db.commit()
            finally:
                db.close()
        return access_token
    except Exception as exc:  # noqa: BLE001
        logger.warning("token refresh failed: %s", exc)
        return None


async def _authed_headers(integration: Integration) -> dict[str, str] | None:
    creds = _get_tokens(integration)
    access = creds.get("access_token")
    return {"Authorization": f"Bearer {access}"}


async def gmail_search(integration: Integration, query: str = "", max_results: int = 10) -> list[dict[str, Any]]:
    """Search Gmail messages. query examples: 'from:vendor subject:invoice', 'company:acme'."""
    import httpx

    headers = await _authed_headers(integration)
    if not headers:
        return []
    params = {"q": query, "maxResults": max_results}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        resp = await client.get("https://gmail.googleapis.com/gmail/v1/users/me/messages", params=params)
        if resp.status_code == 401:
            new_token = await _refresh_access_token(integration)
            if new_token:
                headers = {"Authorization": f"Bearer {new_token}"}
                resp = await client.get("https://gmail.googleapis.com/gmail/v1/users/me/messages", params=params, headers=headers)
            else:
                return []
        resp.raise_for_status()
        data = resp.json()
    messages = data.get("messages", [])[:max_results]
    results = []
    for msg in messages:
        msg_id = msg.get("id")
        detail = await gmail_get_message(integration, msg_id, headers=headers)
        if detail:
            results.append(detail)
    return results


async def gmail_get_message(integration: Integration, message_id: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Fetch a single Gmail message with parsed fields."""
    import httpx

    if not headers:
        headers = await _authed_headers(integration)
    if not headers:
        return {}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        resp = await client.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}", params={"format": "full"})
        resp.raise_for_status()
        data = resp.json()

    payload = data.get("payload", {})
    headers_map = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}
    body = _extract_message_body(payload)
    return {
        "id": message_id,
        "from": headers_map.get("from", ""),
        "to": headers_map.get("to", ""),
        "subject": headers_map.get("subject", ""),
        "date": headers_map.get("date", ""),
        "snippet": data.get("snippet", ""),
        "body": body[:5000],
    }


def _extract_message_body(payload: dict[str, Any]) -> str:
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        try:
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    # recurse into parts
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            if part.get("body", {}).get("data"):
                try:
                    return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                except Exception:  # noqa: BLE001
                    return ""
        text = _extract_message_body(part)
        if text:
            return text
    return ""


async def calendar_upcoming(integration: Integration, max_results: int = 10, days: int = 7) -> list[dict[str, Any]]:
    """Upcoming calendar events from the user's primary calendar."""
    import httpx

    headers = await _authed_headers(integration)
    if not headers:
        return []
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).isoformat()
    end = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    params = {"maxResults": max_results, "timeMin": now, "timeMax": end, "singleEvents": True, "orderBy": "startTime"}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        resp = await client.get("https://www.googleapis.com/calendar/v3/calendars/primary/events", params=params)
        resp.raise_for_status()
        data = resp.json()
    events = []
    for item in data.get("items", []):
        events.append(
            {
                "summary": item.get("summary", ""),
                "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                "location": item.get("location", ""),
                "description": (item.get("description") or "")[:500],
            }
        )
    return events


async def sheets_values(integration: Integration, spreadsheet_id: str, range_name: str = "A1:Z100") -> list[list[Any]]:
    """Read values from a Google Sheet."""
    import httpx

    headers = await _authed_headers(integration)
    if not headers:
        return []
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return data.get("values", [])


async def drive_find_by_name(integration: Integration, query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search Google Drive for files matching a query."""
    import httpx

    headers = await _authed_headers(integration)
    if not headers:
        return []
    full_query = f"name contains '{query}' and trashed = false"
    params = {"q": full_query, "pageSize": max_results, "fields": "files(id,name,mimeType,modifiedTime)"}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        resp = await client.get("https://www.googleapis.com/drive/v3/files", params=params)
        resp.raise_for_status()
        data = resp.json()
    return [
        {
            "id": f.get("id"),
            "name": f.get("name"),
            "mimeType": f.get("mimeType"),
            "modifiedTime": f.get("modifiedTime"),
        }
        for f in data.get("files", [])
    ]


async def drive_read_file_content(
    integration: Integration,
    file_id: str,
    mime_type: str = "",
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Read text content of a Drive file (Google Doc/Sheets/PDF/TXT)."""
    import httpx

    headers = await _authed_headers(integration)
    if not headers:
        return {"error": "Not connected"}

    try:
        if "sheet" in mime_type:
            values = await sheets_values(integration, file_id, "A1:H200")
            text = "\n".join("\t".join(str(c) for c in row) for row in values)
            return {"content": text[:max_chars], "mode": "sheet", "rows": len(values)}
        if "document" in mime_type:
            url = f"https://docs.googleapis.com/v1/documents/{file_id}"
            async with httpx.AsyncClient(headers=headers, timeout=20) as client:
                resp = await client.get(url)
                if resp.status_code == 401:
                    new_token = await _refresh_access_token(integration)
                    if new_token:
                        resp = await client.get(url, headers={"Authorization": f"Bearer {new_token}"})
                resp.raise_for_status()
                doc = resp.json()
            body = doc.get("body", {}).get("content", [])
            chunks = []
            for el in body:
                if el.get("paragraph"):
                    for run in el["paragraph"].get("elements", []):
                        tr = run.get("textRun", {}).get("content", "")
                        if tr:
                            chunks.append(tr)
            return {"content": "".join(chunks)[:max_chars], "mode": "doc"}
        # Plain export for PDF/TXT/etc.
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
        params = {"mimeType": "text/plain"}
        async with httpx.AsyncClient(headers=headers, timeout=30) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 401:
                new_token = await _refresh_access_token(integration)
                if new_token:
                    resp = await client.get(url, params=params, headers={"Authorization": f"Bearer {new_token}"})
            resp.raise_for_status()
            return {"content": resp.text[:max_chars], "mode": "text"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("drive read failed: %s", exc)
        return {"error": str(exc)}


async def calendar_create_event(
    integration: Integration,
    summary: str,
    start_dt: str,
    end_dt: str,
    description: str = "",
    location: str = "",
) -> dict[str, Any]:
    """Create a calendar event. start_dt/end_dt are ISO-8601 datetimes."""
    import httpx

    headers = await _authed_headers(integration)
    if not headers:
        return {"error": "Not connected"}
    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_dt},
        "end": {"dateTime": end_dt},
    }
    if location:
        body["location"] = location
    url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        resp = await client.post(url, json=body)
        if resp.status_code == 401:
            new_token = await _refresh_access_token(integration)
            if new_token:
                resp = await client.post(url, json=body, headers={"Authorization": f"Bearer {new_token}"})
        if resp.status_code not in (200, 201):
            try:
                detail = resp.json()
            except Exception:  # noqa: BLE001
                detail = resp.text
            return {"error": f"Calendar API error {resp.status_code}: {detail}"}
        data = resp.json()
    return {
        "id": data.get("id"),
        "summary": data.get("summary", ""),
        "htmlLink": data.get("htmlLink", ""),
        "start": data.get("start", {}).get("dateTime", ""),
        "end": data.get("end", {}).get("dateTime", ""),
    }


def sheets_analyze(values: list[list[Any]], sheet_name: str = "Sheet") -> dict[str, Any]:
    """Summarize spreadsheet values: headers, row count, numeric columns, outliers."""
    if not values:
        return {"error": "The sheet is empty."}
    headers = values[0]
    rows = values[1:]
    numeric_cols: dict[str, list[float]] = {}
    for row in rows:
        for i, cell in enumerate(row):
            if i >= len(headers):
                continue
            try:
                numeric_cols.setdefault(headers[i], []).append(float(cell))
            except (TypeError, ValueError):
                continue

    stats = {}
    anomalies: list[str] = []
    for col, nums in numeric_cols.items():
        if not nums:
            continue
        mean = sum(nums) / len(nums)
        std = (sum((x - mean) ** 2 for x in nums) / len(nums)) ** 0.5
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        median = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        outliers: set[float] = set()
        # 2σ rule — works well once we have enough rows
        if std > 0:
            outliers |= {x for x in nums if abs(x - mean) > 2 * std}
        # Robust median-ratio rule — catches extreme spikes in small samples
        # where a single outlier inflates the std and masks itself.
        if median > 0:
            outliers |= {x for x in nums if x > 4 * median or x < median / 4}
        stats[col] = {
            "mean": round(mean, 4),
            "median": round(median, 4),
            "min": round(min(nums), 4),
            "max": round(max(nums), 4),
        }
        if outliers:
            anomalies.append(f"{col}: unusual values {[round(o, 4) for o in sorted(outliers)[:5]]}")

    return {
        "sheet": sheet_name,
        "headers": headers,
        "row_count": len(rows),
        "numeric_stats": stats,
        "anomalies": anomalies,
    }


def describe_connection_status(user: User) -> str:
    """Human-readable summary of a user's connected integrations."""
    if not user.integrations:
        return "No integrations connected yet."
    lines = []
    labels = {
        "gmail": "📧 Gmail",
        "google_calendar": "📅 Google Calendar",
        "google_sheets": "📊 Google Sheets",
        "google_drive": "🗂 Google Drive",
    }
    for integration in user.integrations:
        label = labels.get(integration.provider, integration.provider)
        lines.append(label)
    return "Connected: " + ", ".join(lines)

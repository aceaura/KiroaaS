use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;

use crate::config::{get_app_data_dir, load_config, save_config};

const DEFAULT_AUTH_BASE: &str = "https://cloud.kiroaas.hnew.city";
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const CLIENT_PLATFORM: &str = std::env::consts::OS;

fn auth_base_url() -> String {
    std::env::var("KIROAAS_CLOUD_AUTH_URL").unwrap_or_else(|_| DEFAULT_AUTH_BASE.to_string())
}

fn session_file_path() -> Result<PathBuf, String> {
    Ok(get_app_data_dir()?.join("cloud_session.json"))
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct StoredSession {
    email: String,
    token: String,
    #[serde(rename = "expiresAt")]
    expires_at: i64,
}

#[derive(Serialize)]
struct Credentials {
    email: String,
    password: String,
}

#[derive(Deserialize)]
struct AuthResponse {
    token: String,
    email: String,
    #[serde(rename = "expiresAt")]
    expires_at: i64,
}

#[derive(Deserialize)]
struct ApiError {
    error: String,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CloudSession {
    pub email: String,
    pub has_token: bool,
    pub expires_at: Option<i64>,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CloudPingResult {
    pub ok: bool,
    pub email: Option<String>,
    pub expires_at: Option<i64>,
    pub status: u16,
    pub error: Option<String>,
    pub latency_ms: Option<u64>,
}

/// Read client_id from config, generating + persisting one on first call.
async fn ensure_client_id() -> String {
    let mut config = match load_config().await {
        Ok(c) => c,
        Err(_) => return uuid::Uuid::new_v4().to_string(),
    };
    if let Some(id) = config.client_id.as_ref() {
        if !id.is_empty() {
            return id.clone();
        }
    }
    let new_id = uuid::Uuid::new_v4().to_string();
    config.client_id = Some(new_id.clone());
    let _ = save_config(&config).await;
    new_id
}

async fn post_auth(path: &str, creds: &Credentials) -> Result<AuthResponse, String> {
    let url = format!("{}/auth/{}", auth_base_url(), path);
    let resp = reqwest::Client::new()
        .post(&url)
        .json(creds)
        .send()
        .await
        .map_err(|e| format!("network error: {}", e))?;
    let status = resp.status();
    if status.is_success() {
        resp.json::<AuthResponse>()
            .await
            .map_err(|e| format!("decode error: {}", e))
    } else {
        let err = resp.json::<ApiError>().await.ok();
        Err(err.map(|e| e.error).unwrap_or_else(|| format!("HTTP {}", status)))
    }
}

async fn save_session(stored: &StoredSession) -> Result<(), String> {
    let path = session_file_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("failed to create data dir: {}", e))?;
    }
    let json = serde_json::to_string(stored).map_err(|e| e.to_string())?;
    fs::write(&path, json).await.map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

async fn load_session() -> Option<StoredSession> {
    let path = session_file_path().ok()?;
    let content = fs::read_to_string(&path).await.ok()?;
    serde_json::from_str(&content).ok()
}

async fn clear_session() {
    if let Ok(path) = session_file_path() {
        let _ = fs::remove_file(&path).await;
    }
}

/// Read the cached token synchronously for spawning the Python backend.
/// Used by server.rs at startup; runs off-Tokio so we use std::fs.
pub fn read_token_blocking() -> Option<String> {
    let path = session_file_path().ok()?;
    let content = std::fs::read_to_string(&path).ok()?;
    let stored: StoredSession = serde_json::from_str(&content).ok()?;
    if stored.token.is_empty() { None } else { Some(stored.token) }
}

#[tauri::command]
pub async fn cloud_register(email: String, password: String) -> Result<CloudSession, String> {
    let resp = post_auth("register", &Credentials { email, password }).await?;
    save_session(&StoredSession {
        email: resp.email.clone(),
        token: resp.token,
        expires_at: resp.expires_at,
    })
    .await?;
    Ok(CloudSession { email: resp.email, has_token: true, expires_at: Some(resp.expires_at) })
}

#[tauri::command]
pub async fn cloud_login(email: String, password: String) -> Result<CloudSession, String> {
    let resp = post_auth("login", &Credentials { email, password }).await?;
    save_session(&StoredSession {
        email: resp.email.clone(),
        token: resp.token,
        expires_at: resp.expires_at,
    })
    .await?;
    Ok(CloudSession { email: resp.email, has_token: true, expires_at: Some(resp.expires_at) })
}

#[tauri::command]
pub async fn cloud_logout() -> Result<(), String> {
    let token = load_session().await.map(|s| s.token);
    clear_session().await;
    if let Some(token) = token {
        if !token.is_empty() {
            let url = format!("{}/auth/logout", auth_base_url());
            let _ = reqwest::Client::new()
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .send()
                .await;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn cloud_get_session() -> CloudSession {
    match load_session().await {
        Some(s) if !s.token.is_empty() => CloudSession {
            email: s.email,
            has_token: true,
            expires_at: Some(s.expires_at),
        },
        _ => CloudSession::default(),
    }
}

/// Hit /auth/me to verify the cached token is still alive on the cloud side.
/// Sends X-Client-Id / X-Client-Version / X-Client-Platform so the backend
/// can stamp the user's row with last-active info.
#[tauri::command]
pub async fn cloud_ping() -> CloudPingResult {
    let token = match load_session().await {
        Some(s) if !s.token.is_empty() => s.token,
        _ => {
            return CloudPingResult {
                ok: false,
                status: 401,
                error: Some("not signed in".into()),
                ..Default::default()
            };
        }
    };
    let client_id = ensure_client_id().await;
    let url = format!("{}/auth/me", auth_base_url());
    let start = std::time::Instant::now();
    let resp = reqwest::Client::new()
        .get(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("X-Client-Id", client_id)
        .header("X-Client-Version", CLIENT_VERSION)
        .header("X-Client-Platform", CLIENT_PLATFORM)
        .send()
        .await;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            if r.status().is_success() {
                #[derive(Deserialize)]
                struct MeResp {
                    email: String,
                    #[serde(rename = "expiresAt")]
                    expires_at: Option<i64>,
                }
                match r.json::<MeResp>().await {
                    Ok(me) => CloudPingResult {
                        ok: true,
                        email: Some(me.email),
                        expires_at: me.expires_at,
                        status,
                        error: None,
                        latency_ms: Some(elapsed_ms),
                    },
                    Err(e) => CloudPingResult {
                        ok: false,
                        status,
                        error: Some(format!("decode error: {}", e)),
                        latency_ms: Some(elapsed_ms),
                        ..Default::default()
                    },
                }
            } else {
                if status == 401 {
                    clear_session().await;
                }
                let err = r.json::<ApiError>().await.ok().map(|e| e.error);
                CloudPingResult {
                    ok: false,
                    status,
                    error: err.or(Some(format!("HTTP {}", status))),
                    latency_ms: Some(elapsed_ms),
                    ..Default::default()
                }
            }
        }
        Err(e) => CloudPingResult {
            ok: false,
            status: 0,
            error: Some(format!("network error: {}", e)),
            latency_ms: Some(elapsed_ms),
            ..Default::default()
        },
    }
}

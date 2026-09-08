import { LATEST_RELEASE_API } from './config';
import { getAppVersion } from './tauri';

export interface UpdateInfo {
  hasUpdate: boolean;
  latestVersion?: string;
  currentVersion?: string;
  downloadUrl?: string;
  changelog?: Array<{ version: string; changes: string[] }>;
}

interface GitHubRelease {
  tag_name: string;
  html_url: string;
  body?: string | null;
}

function parseVersion(v: string): number[] {
  return v
    .replace(/^v/i, '')
    .split('.')
    .map((part) => parseInt(part, 10) || 0);
}

function isNewerVersion(latest: string, current: string): boolean {
  const a = parseVersion(latest);
  const b = parseVersion(current);
  const len = Math.max(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    if (diff !== 0) return diff > 0;
  }
  return false;
}

export async function checkVersionUpdate(): Promise<UpdateInfo | null> {
  try {
    const currentVersion = await getAppVersion();

    const response = await fetch(LATEST_RELEASE_API, {
      headers: { Accept: 'application/vnd.github+json' },
    });
    if (!response.ok) {
      throw new Error(`GitHub API responded ${response.status}`);
    }
    const release: GitHubRelease = await response.json();

    const latestVersion = release.tag_name.replace(/^v/i, '');
    const changes = (release.body ?? '')
      .split('\n')
      .map((line) => line.trim().replace(/^[-*]\s+/, ''))
      .filter(Boolean)
      .slice(0, 30);

    const info: UpdateInfo = {
      hasUpdate: isNewerVersion(latestVersion, currentVersion),
      latestVersion,
      currentVersion,
      downloadUrl: release.html_url,
      changelog: changes.length ? [{ version: latestVersion, changes }] : undefined,
    };
    console.log('[CheckUpdate]', info);
    return info;
  } catch (err) {
    console.error('[CheckUpdate] Failed:', err);
    return null;
  }
}

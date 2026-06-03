import { useEffect, useRef, useState } from 'react';
import { Cloud, LogOut, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { useI18n } from '@/hooks/useI18n';
import {
  cloudGetSession,
  cloudLogin,
  cloudLogout,
  cloudPing,
  cloudRegister,
  type CloudSession,
} from '@/lib/tauri';

interface CloudViewProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  onAuthChanged?: () => void;
}

type PingStatus = 'idle' | 'checking' | 'connected' | 'expired' | 'unreachable';

const PING_INTERVAL_MS = 10 * 60 * 1000;

export function CloudView({ enabled, onEnabledChange, onAuthChanged }: CloudViewProps) {
  const { t } = useI18n();
  const [session, setSession] = useState<CloudSession>({ email: '', hasToken: false });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [pingStatus, setPingStatus] = useState<PingStatus>('idle');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const pingTimerRef = useRef<number | null>(null);
  const onAuthChangedRef = useRef(onAuthChanged);

  useEffect(() => {
    onAuthChangedRef.current = onAuthChanged;
  }, [onAuthChanged]);

  useEffect(() => {
    cloudGetSession().then(setSession).catch(() => {});
  }, []);

  // Periodic ping while signed in: confirms cloud reachability and stamps
  // last-active on the user row. Pauses when the window is hidden so we
  // don't burn lambda invocations on a backgrounded app.
  useEffect(() => {
    if (!session.hasToken) {
      setPingStatus('idle');
      setLatencyMs(null);
      return;
    }

    let cancelled = false;
    const runPing = async () => {
      if (cancelled) return;
      setPingStatus((prev) => (prev === 'connected' ? prev : 'checking'));
      const result = await cloudPing();
      if (cancelled) return;
      setLatencyMs(result.latencyMs ?? null);
      if (result.ok) {
        setPingStatus('connected');
      } else if (result.status === 401) {
        setPingStatus('expired');
        setSession({ email: '', hasToken: false });
        onAuthChangedRef.current?.();
      } else {
        setPingStatus('unreachable');
      }
    };

    const start = () => {
      if (pingTimerRef.current != null) return;
      runPing();
      pingTimerRef.current = window.setInterval(runPing, PING_INTERVAL_MS);
    };
    const stop = () => {
      if (pingTimerRef.current != null) {
        window.clearInterval(pingTimerRef.current);
        pingTimerRef.current = null;
      }
    };
    const onVisibility = () => {
      if (document.hidden) {
        stop();
      } else {
        start();
      }
    };

    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelled = true;
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [session.hasToken]);

  const submit = async () => {
    if (mode === 'register' && password !== confirmPassword) {
      setError(t('cloudPasswordMismatch'));
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const fn = mode === 'login' ? cloudLogin : cloudRegister;
      const next = await fn(email.trim(), password);
      setSession(next);
      setPassword('');
      setConfirmPassword('');
      onAuthChanged?.();
    } catch (e) {
      setError(typeof e === 'string' ? e : (e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (next: 'login' | 'register') => {
    setMode(next);
    setError(null);
    setPassword('');
    setConfirmPassword('');
  };

  const handleLogout = async () => {
    setLoading(true);
    try {
      await cloudLogout();
      setSession({ email: '', hasToken: false });
      if (enabled) {
        onEnabledChange(false);
      }
      onAuthChanged?.();
    } finally {
      setLoading(false);
    }
  };

  const showLoginNeededHint = enabled && !session.hasToken;

  const statusDot = (() => {
    switch (pingStatus) {
      case 'connected':
        return { color: 'bg-lime-500', label: t('cloudStatusConnected'), pulse: false };
      case 'checking':
        return { color: 'bg-amber-400', label: t('cloudStatusChecking'), pulse: true };
      case 'expired':
        return { color: 'bg-red-500', label: t('cloudStatusExpired'), pulse: false };
      case 'unreachable':
        return { color: 'bg-stone-400', label: t('cloudStatusUnreachable'), pulse: false };
      default:
        return null;
    }
  })();

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="bg-white rounded-2xl p-6 shadow-sm">
        <div className="flex items-start justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Cloud className="h-5 w-5 text-stone-700" />
              <h2 className="font-semibold text-[#111]">{t('cloudEnableTitle')}</h2>
            </div>
            <p className="text-sm text-stone-500">{t('cloudEnableDesc')}</p>
            <p className="text-xs text-stone-400 mt-1.5">{t('cloudPrivacyNote')}</p>
          </div>
          <Switch
            checked={enabled}
            onCheckedChange={onEnabledChange}
            disabled={!session.hasToken && !enabled}
          />
        </div>

        {showLoginNeededHint && (
          <div className="mt-4 flex items-start gap-2 rounded-xl bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-800">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <span>{t('cloudLoginNeeded')}</span>
          </div>
        )}
      </div>

      <div className="bg-white rounded-2xl p-6 shadow-sm">
        {session.hasToken ? (
          <>
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-lime-100 flex items-center justify-center">
                  <CheckCircle2 className="h-5 w-5 text-lime-700" />
                </div>
                <div>
                  <p className="text-sm text-stone-500">{t('cloudSignedIn')}</p>
                  <p className="font-semibold text-[#111]">{session.email || t('cloudUnknownAccount')}</p>
                </div>
              </div>
              <Button
                variant="outline"
                onClick={handleLogout}
                disabled={loading}
                className="rounded-full"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogOut className="h-4 w-4 mr-2" />}
                {t('cloudLogoutBtn')}
              </Button>
            </div>
            {statusDot && (
              <div className="mt-4 flex items-center gap-2 text-xs text-stone-500">
                <span className="relative flex h-2 w-2">
                  {statusDot.pulse && (
                    <span className={`absolute inline-flex h-full w-full rounded-full ${statusDot.color} opacity-75 animate-ping`} />
                  )}
                  <span className={`relative inline-flex h-2 w-2 rounded-full ${statusDot.color}`} />
                </span>
                <span>{statusDot.label}</span>
                {pingStatus === 'connected' && latencyMs != null && (
                  <span className="ml-1 font-mono tabular-nums text-stone-400">
                    {latencyMs}ms
                  </span>
                )}
              </div>
            )}
          </>
        ) : (
          <div>
            <div className="mb-5">
              <div className="h-10 w-10 rounded-xl bg-[#111] flex items-center justify-center mb-4">
                <Cloud className="h-5 w-5 text-white" />
              </div>
              <h2 className="text-xl font-semibold text-[#111] mb-2">
                {mode === 'login' ? t('cloudCardTitleLogin') : t('cloudCardTitleRegister')}
              </h2>
              <p className="text-sm text-stone-500 leading-relaxed">
                {t('cloudCardDesc')}
              </p>
              <p className="mt-2 text-xs text-stone-400 leading-relaxed">
                {t('cloudCardPricingNote')}
              </p>
            </div>

            <form
              className="space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <div className="space-y-1.5">
                <Label htmlFor="cloud-email" className="sr-only">{t('cloudFieldEmail')}</Label>
                <Input
                  id="cloud-email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder={t('cloudFieldEmail')}
                  className="h-11 rounded-xl"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="cloud-password" className="sr-only">{t('cloudFieldPassword')}</Label>
                <Input
                  id="cloud-password"
                  type="password"
                  autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  required
                  minLength={8}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={mode === 'register' ? t('cloudPasswordHint') : t('cloudFieldPassword')}
                  className="h-11 rounded-xl"
                />
              </div>

              {mode === 'register' && (
                <div className="space-y-1.5">
                  <Label htmlFor="cloud-password-confirm" className="sr-only">{t('cloudFieldConfirmPassword')}</Label>
                  <Input
                    id="cloud-password-confirm"
                    type="password"
                    autoComplete="new-password"
                    required
                    minLength={8}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder={t('cloudFieldConfirmPassword')}
                    className="h-11 rounded-xl"
                  />
                </div>
              )}

              {error && (
                <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
                  <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <Button
                type="submit"
                disabled={loading || !email || !password || (mode === 'register' && !confirmPassword)}
                className="w-full h-11 rounded-xl bg-lime-400 text-[#111] font-semibold hover:bg-lime-500 disabled:opacity-60 disabled:bg-lime-400"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                {mode === 'login' ? t('cloudSubmitLogin') : t('cloudSubmitRegister')}
              </Button>
            </form>

            <div className="mt-5 pt-5 border-t border-stone-100 text-sm text-stone-500">
              {mode === 'login' ? (
                <>
                  {t('cloudNoAccount')}{' '}
                  <button
                    type="button"
                    onClick={() => switchMode('register')}
                    className="font-semibold text-[#111] hover:underline"
                  >
                    {t('cloudGoToRegister')}
                  </button>
                </>
              ) : (
                <>
                  {t('cloudHaveAccount')}{' '}
                  <button
                    type="button"
                    onClick={() => switchMode('login')}
                    className="font-semibold text-[#111] hover:underline"
                  >
                    {t('cloudGoToLogin')}
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

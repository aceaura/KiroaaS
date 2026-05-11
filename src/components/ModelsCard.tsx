import { useState, useEffect, useCallback, useRef } from 'react';
import { Sparkles, Loader2, ChevronDown, Copy, Check } from 'lucide-react';
import { useI18n } from '@/hooks/useI18n';

interface ModelsCardProps {
  host: string;
  port: number;
  apiKey: string;
  isRunning: boolean;
}

interface OpenAIModel {
  id: string;
  description?: string;
}

interface ModelList {
  data?: OpenAIModel[];
}

let cachedModels: ModelList | null = null;

export function ModelsCard({ host, port, apiKey, isRunning }: ModelsCardProps) {
  const { t } = useI18n();
  const [models, setModels] = useState<ModelList | null>(cachedModels);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const copyId = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    navigator.clipboard?.writeText(id).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId((curr) => (curr === id ? null : curr)), 1200);
    });
  };

  const fetchModels = useCallback(async () => {
    if (!isRunning || !apiKey) return;
    if (!cachedModels) setLoading(true);
    try {
      const fetchHost = host === '0.0.0.0' ? '127.0.0.1' : host;
      const res = await fetch(`http://${fetchHost}:${port}/v1/models`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      if (!res.ok) return;
      const data: ModelList = await res.json();
      setModels(data);
      cachedModels = data;
    } catch {
      // Silently ignore — secondary info.
    } finally {
      setLoading(false);
    }
  }, [host, port, apiKey, isRunning]);

  useEffect(() => {
    if (!isRunning) {
      setModels(null);
      cachedModels = null;
      setExpanded(false);
      return;
    }
    fetchModels();
    const interval = setInterval(fetchModels, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [isRunning, fetchModels]);

  const list = (models?.data ?? []).filter(
    (m) => !m.id.toLowerCase().startsWith('auto')
  );
  const count = list.length;
  const canToggle = isRunning && count > 0;

  const toggle = () => {
    if (canToggle) setExpanded((v) => !v);
  };

  return (
    <div className="relative flex-1 min-h-[92px]">
      <div
        className={`absolute inset-x-0 top-0 bg-[#111] text-white border-0 rounded-[32px] overflow-hidden origin-top transition-all duration-300 ease-out ${
          canToggle ? 'cursor-pointer' : ''
        } ${
          expanded
            ? 'shadow-2xl shadow-black/40 z-20 scale-[1.02] ring-1 ring-white/10'
            : 'shadow-sm hover:shadow-lg hover:bg-[#1a1a1a]'
        }`}
        onClick={toggle}
      >
        <div className="absolute top-0 right-0 w-24 h-24 bg-stone-800 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2 opacity-50 pointer-events-none" />

        {/* Header (always visible) */}
        <div className="px-6 py-4 flex flex-col justify-center min-h-[92px] relative z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="h-3 w-3 text-stone-400" />
              <span className="text-stone-400 font-semibold text-[10px] tracking-wider uppercase">
                {t('supportedModels')}
              </span>
            </div>
            {canToggle && (
              <ChevronDown
                className={`h-4 w-4 text-stone-400 transition-transform duration-300 ${
                  expanded ? 'rotate-180' : ''
                }`}
              />
            )}
          </div>
          <div className="flex items-baseline gap-2 mt-1">
            {loading && !models ? (
              <Loader2 className="h-5 w-5 animate-spin text-stone-400" />
            ) : (
              <>
                <span className="text-2xl font-bold tracking-tight">
                  {isRunning ? count : '—'}
                </span>
                {isRunning && count > 0 && !expanded && (
                  <span className="text-[10px] text-stone-400 font-medium">
                    {t('modelsClickToView')}
                  </span>
                )}
              </>
            )}
          </div>
        </div>

        {/* Collapsible body */}
        <div
          className="grid transition-[grid-template-rows] duration-300 ease-out relative z-10"
          style={{ gridTemplateRows: expanded ? '1fr' : '0fr' }}
        >
          <div className="overflow-hidden">
            <div
              ref={listRef}
              className="px-6 pb-5 pt-1 space-y-1.5 max-h-[320px] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {list.map((m, i) => (
                <div
                  key={m.id}
                  className={`group flex items-center gap-2 px-3 py-1 bg-white/5 hover:bg-white/10 border border-white/5 rounded-lg transition-all ${
                    expanded
                      ? 'opacity-100 translate-y-0'
                      : 'opacity-0 -translate-y-1'
                  }`}
                  style={{
                    transitionDuration: '250ms',
                    transitionDelay: expanded ? `${Math.min(i * 20, 200)}ms` : '0ms',
                  }}
                  title={m.description || m.id}
                >
                  <span className="text-xs font-mono text-white truncate flex-1 min-w-0">{m.id}</span>
                  <button
                    type="button"
                    onClick={(e) => copyId(e, m.id)}
                    aria-label="Copy"
                    className={`h-6 w-6 rounded-md flex items-center justify-center shrink-0 transition-all ${
                      copiedId === m.id
                        ? 'bg-lime-500/20 text-lime-300 opacity-100'
                        : 'text-stone-400 hover:text-white hover:bg-white/10 opacity-0 group-hover:opacity-100'
                    }`}
                  >
                    {copiedId === m.id ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

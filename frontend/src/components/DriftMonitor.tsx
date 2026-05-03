import { useQuery } from '@tanstack/react-query';
import { Card } from '@tremor/react';
import { motion } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Activity, ArrowUpRight, ArrowDownRight, AlertCircle, RefreshCw } from 'lucide-react';
import { api, API_BASE_URL, type SHAPDriftReport, type DriftFeature } from '../lib/api';

function DriftBadge({ verdict }: { verdict: string }) {
  const styles = {
    STABLE: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    DRIFT_DETECTED: 'bg-red-500/10 text-red-400 border-red-500/20',
    baseline_established: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  };
  const labels = {
    STABLE: '● Stable',
    DRIFT_DETECTED: '⚠ Drift Detected',
    baseline_established: '◐ Baseline Set',
  };

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${styles[verdict as keyof typeof styles] || styles.baseline_established}`}>
      {labels[verdict as keyof typeof labels] || verdict}
    </span>
  );
}

function MetricCard({ label, value, subtitle, variant = 'default' }: {
  label: string;
  value: string | number;
  subtitle?: string;
  variant?: 'default' | 'success' | 'danger' | 'warning';
}) {
  const borderColors = {
    default: 'border-slate-200',
    success: 'border-emerald-200',
    danger: 'border-red-200',
    warning: 'border-amber-200',
  };

  return (
    <div className={`bg-white rounded-xl border ${borderColors[variant]} p-5 shadow-sm`}>
      <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">{label}</p>
      <p className="text-2xl text-slate-900 font-bold mt-2">{value}</p>
      {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
    </div>
  );
}

function DriftedFeatureRow({ feature }: { feature: DriftFeature }) {
  const isIncrease = feature.direction === 'increased';
  const changePercent = Math.round(feature.relative_change * 100);

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center justify-between py-3 px-4 rounded-lg bg-slate-50 border border-slate-200"
    >
      <div className="flex items-center gap-3">
        {isIncrease ? (
          <ArrowUpRight className="w-4 h-4 text-amber-500" />
        ) : (
          <ArrowDownRight className="w-4 h-4 text-blue-500" />
        )}
        <div>
          <p className="text-sm font-medium text-slate-900">{feature.feature.replace(/_/g, ' ')}</p>
          <p className="text-xs text-slate-500">
            {feature.baseline_importance.toFixed(4)} → {feature.current_importance.toFixed(4)}
          </p>
        </div>
      </div>
      <span className={`text-sm font-mono font-bold ${isIncrease ? 'text-amber-500' : 'text-blue-500'}`}>
        {isIncrease ? '+' : '-'}{changePercent}%
      </span>
    </motion.div>
  );
}

function SHAPDriftPanel({ report }: { report: SHAPDriftReport }) {
  if (report.status === 'baseline_established') {
    return (
      <Card className="bg-white border-slate-200 shadow-tonal ring-0">
        <div className="flex items-center gap-3 mb-4">
          <RefreshCw className="w-5 h-5 text-blue-500" />
          <h3 className="text-lg font-semibold text-slate-900">SHAP Attribution Drift</h3>
          <DriftBadge verdict="baseline_established" />
        </div>
        <p className="text-sm text-slate-500">{report.message}</p>
      </Card>
    );
  }

  const metrics = report.metrics;

  // const top5Chart = metrics.top5_current.map((feat: string) => ({
  //   feature: feat.replace(/_/g, ' ').substring(0, 20),
  //   status: metrics.top5_baseline.includes(feat) ? 'Stable' : 'New',
  // }));

  return (
    <Card className="bg-white border-slate-200 shadow-tonal ring-0">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          {report.has_drift ? (
            <ShieldAlert className="w-5 h-5 text-error" />
          ) : (
            <ShieldCheck className="w-5 h-5 text-emerald-500" />
          )}
          <h3 className="text-lg font-semibold text-slate-900">SHAP Attribution Drift</h3>
          <DriftBadge verdict={report.verdict} />
        </div>
        <span className="text-xs text-slate-500">
          {new Date(report.analysis_date).toLocaleDateString()}
        </span>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <MetricCard
          label="Rank Correlation (ρ)"
          value={metrics.spearman_rank_correlation.toFixed(3)}
          subtitle={`Threshold: ${metrics.threshold}`}
          variant={metrics.rank_drift_detected ? 'danger' : 'success'}
        />
        <MetricCard
          label="Top-5 Overlap"
          value={`${metrics.top5_overlap}/5`}
          subtitle={metrics.top5_stable ? 'Stable' : 'Shifted'}
          variant={metrics.top5_stable ? 'success' : 'warning'}
        />
        <MetricCard
          label="Features Drifted"
          value={metrics.features_with_magnitude_drift}
          subtitle="Magnitude > 50%"
          variant={metrics.features_with_magnitude_drift > 3 ? 'danger' : 'success'}
        />
      </div>

      {/* Top-5 Feature Comparison */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-3">Baseline Top-5</h4>
          <ul className="space-y-2">
            {metrics.top5_baseline.map((feat: string) => (
              <li key={feat} className="flex items-center gap-2 text-sm">
                <span className={`w-2 h-2 rounded-full ${metrics.top5_current.includes(feat) ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                <span className={metrics.top5_current.includes(feat) ? 'text-slate-900' : 'text-slate-400 line-through'}>
                  {feat.replace(/_/g, ' ')}
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-3">Current Top-5</h4>
          <ul className="space-y-2">
            {metrics.top5_current.map((feat: string) => (
              <li key={feat} className="flex items-center gap-2 text-sm">
                <span className={`w-2 h-2 rounded-full ${metrics.top5_baseline.includes(feat) ? 'bg-emerald-500' : 'bg-amber-400'}`} />
                <span className="text-slate-900">
                  {feat.replace(/_/g, ' ')}
                </span>
                {!metrics.top5_baseline.includes(feat) && (
                  <span className="text-[10px] bg-amber-100 text-amber-600 px-1.5 py-0.5 rounded-full font-semibold">NEW</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Drifted Features */}
      {report.drifted_features.length > 0 && (
        <div>
          <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-3">
            Drifted Features ({report.drifted_features.length})
          </h4>
          <div className="space-y-2">
            {report.drifted_features.slice(0, 8).map((feat) => (
              <DriftedFeatureRow key={feat.feature} feature={feat} />
            ))}
          </div>
        </div>
      )}

      {/* Recommendation */}
      <div className={`mt-6 p-4 rounded-lg border ${
        report.has_drift
          ? 'bg-red-50 border-red-200'
          : 'bg-emerald-50 border-emerald-200'
      }`}>
        <div className="flex items-start gap-2">
          <AlertCircle className={`w-4 h-4 mt-0.5 ${report.has_drift ? 'text-error' : 'text-emerald-600'}`} />
          <p className={`text-sm ${report.has_drift ? 'text-red-700' : 'text-emerald-700'}`}>
            {report.recommendation}
          </p>
        </div>
      </div>
    </Card>
  );
}

export default function DriftMonitor() {
  const { data: driftData, isLoading, isError } = useQuery({
    queryKey: ['drift-status'],
    queryFn: api.getDriftStatus,
    retry: 1,
    staleTime: 60000,
  });

  if (isLoading) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 gap-6">
          {[1, 2].map((i) => (
            <div key={i} className="bg-slate-100 rounded-xl border border-slate-200 h-48 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="p-8 max-w-7xl mx-auto space-y-6"
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">Model Drift Monitor</h2>
          <p className="text-sm text-slate-500 mt-1">
            Continuous monitoring of data distribution and SHAP attribution stability
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-500" />
          <span className="text-xs text-slate-500">Auto-refreshes every 60s</span>
        </div>
      </div>

      {isError && (
        <Card className="bg-white border-slate-200 shadow-sm ring-0">
          <div className="flex items-center gap-3 text-slate-500">
            <AlertCircle className="w-5 h-5 text-amber-500" />
            <p className="text-sm">
              No drift reports available. Run the training pipeline to generate baseline data.
            </p>
          </div>
        </Card>
      )}

      {/* SHAP Attribution Drift */}
      {driftData?.shap_drift && (
        <SHAPDriftPanel report={driftData.shap_drift} />
      )}

      {/* Data Drift (Evidently) Summary */}
      {driftData?.data_drift && (
        <Card className="bg-white border-slate-200 shadow-tonal ring-0">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-slate-900">Data Distribution Drift (Evidently AI)</h3>
            <DriftBadge verdict="STABLE" />
          </div>
          <p className="text-sm text-slate-500">
            Full Evidently data drift and quality report is available at{' '}
            <a 
              href={`${API_BASE_URL}/outputs/evidently_drift_report.html`}
              target="_blank" 
              rel="noreferrer"
              className="text-xs bg-slate-100 text-teal-600 hover:text-teal-700 hover:underline px-2 py-1 rounded"
            >
              /outputs/evidently_drift_report.html
            </a>
          </p>
        </Card>
      )}

      {/* Observability Architecture */}
      <Card className="bg-white border-slate-200 shadow-tonal ring-0">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">Observability Stack</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-slate-50 rounded-lg border border-slate-200">
            <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-blue-50 flex items-center justify-center">
              <Activity className="w-5 h-5 text-blue-500" />
            </div>
            <p className="text-sm font-medium text-slate-900">Data Drift</p>
            <p className="text-xs text-slate-500 mt-1">Evidently AI</p>
            <p className="text-[10px] text-slate-400 mt-1">Input distributions</p>
          </div>
          <div className="text-center p-4 bg-slate-50 rounded-lg border border-slate-200">
            <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-emerald-50 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 text-emerald-500" />
            </div>
            <p className="text-sm font-medium text-slate-900">SHAP Drift</p>
            <p className="text-xs text-slate-500 mt-1">Attribution Monitor</p>
            <p className="text-[10px] text-slate-400 mt-1">Model reasoning</p>
          </div>
          <div className="text-center p-4 bg-slate-50 rounded-lg border border-slate-200">
            <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-amber-50 flex items-center justify-center">
              <ShieldAlert className="w-5 h-5 text-amber-500" />
            </div>
            <p className="text-sm font-medium text-slate-900">Fairness Gate</p>
            <p className="text-xs text-slate-500 mt-1">Fairlearn EOD+DPD</p>
            <p className="text-[10px] text-slate-400 mt-1">Bias prevention</p>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

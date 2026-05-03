import { useState, useEffect } from 'react';
import { Card } from '@tremor/react';
import { ScrollText, Filter, ChevronLeft, ChevronRight, Clock, User, AlertTriangle, Shield } from 'lucide-react';
import { motion } from 'framer-motion';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface AuditLog {
  id: number;
  timestamp: string;
  employee_id: string | null;
  requester_id: string | null;
  attrition_probability: number;
  risk_tier: string;
  top_risk_drivers: Record<string, number> | null;
  generated_strategy: string | null;
}

interface AuditResponse {
  total_count: number;
  page: number;
  page_size: number;
  logs: AuditLog[];
}

export default function AuditTrail() {
  const [data, setData] = useState<AuditResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [filterTier, setFilterTier] = useState<string>('');
  const [filterEmployee, setFilterEmployee] = useState('');
  const pageSize = 10;

  const fetchLogs = async (p: number) => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        page: p.toString(),
        page_size: pageSize.toString(),
      });
      if (filterTier) params.set('risk_tier', filterTier);
      if (filterEmployee) params.set('employee_id', filterEmployee);

      const res = await fetch(`${API_BASE}/v1/audit-logs?${params}`);
      if (res.ok) {
        const json: AuditResponse = await res.json();
        setData(json);
      }
    } catch (e) {
      console.error('Failed to fetch audit logs:', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs(page);
  }, [page, filterTier]);

  const totalPages = data ? Math.ceil(data.total_count / pageSize) : 0;

  const tierColor = (tier: string) => {
    if (tier === 'High') return 'bg-red-50 text-error border-red-200';
    if (tier === 'Medium') return 'bg-amber-50 text-amber-600 border-amber-200';
    return 'bg-emerald-50 text-emerald-600 border-emerald-200';
  };

  const formatTimestamp = (ts: string) => {
    try {
      const d = new Date(ts);
      return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
    } catch { return ts; }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-8 max-w-7xl mx-auto space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-medium text-slate-900">Prediction Audit Trail</h3>
          <p className="text-sm text-slate-500">
            EU AI Act Art. 12 — Every prediction is logged for compliance and traceability.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-500" />
          <span className="text-xs font-medium text-blue-700 bg-blue-50 px-3 py-1 rounded-full border border-blue-200">
            {data?.total_count ?? 0} Records
          </span>
        </div>
      </div>

      {/* Filters */}
      <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-4">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2 text-slate-500">
            <Filter className="w-4 h-4" />
            <span className="text-xs font-medium uppercase tracking-wider">Filters</span>
          </div>
          
          <select
            id="filter-risk-tier"
            value={filterTier}
            onChange={(e) => { setFilterTier(e.target.value); setPage(1); }}
            className="bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-sm text-slate-900 focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            <option value="">All Risk Tiers</option>
            <option value="High">High Risk</option>
            <option value="Medium">Medium Risk</option>
            <option value="Low">Low Risk</option>
          </select>

          <div className="flex items-center gap-2">
            <input
              id="filter-employee-id"
              type="text"
              placeholder="Employee ID..."
              value={filterEmployee}
              onChange={(e) => setFilterEmployee(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { setPage(1); fetchLogs(1); } }}
              className="bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-teal-500 w-40"
            />
          </div>
        </div>
      </Card>

      {/* Log Table */}
      <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-500 text-xs uppercase tracking-wider">
                <th className="px-6 py-4 text-left font-medium">Timestamp</th>
                <th className="px-6 py-4 text-left font-medium">Employee</th>
                <th className="px-6 py-4 text-left font-medium">Risk Score</th>
                <th className="px-6 py-4 text-left font-medium">Tier</th>
                <th className="px-6 py-4 text-left font-medium">Top Drivers</th>
                <th className="px-6 py-4 text-left font-medium">Requester</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="border-b border-slate-100">
                    {Array.from({ length: 6 }).map((_, j) => (
                      <td key={j} className="px-6 py-4">
                        <div className="h-4 bg-slate-100 rounded animate-pulse w-20" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : data && data.logs.length > 0 ? (
                data.logs.map((log) => (
                  <tr
                    key={log.id}
                    className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2 text-slate-600">
                        <Clock className="w-3.5 h-3.5 text-slate-400" />
                        <span className="text-xs">{formatTimestamp(log.timestamp)}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <User className="w-3.5 h-3.5 text-slate-400" />
                        <span className="text-slate-900 font-mono text-xs">{log.employee_id || 'N/A'}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-slate-900 font-bold text-lg">
                        {(log.attrition_probability * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2.5 py-1 text-xs font-bold rounded-md border ${tierColor(log.risk_tier)}`}>
                        {log.risk_tier}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      {log.top_risk_drivers ? (
                        <div className="space-y-1">
                          {Object.entries(log.top_risk_drivers).slice(0, 2).map(([feat, val]) => (
                            <div key={feat} className="flex items-center gap-2">
                              <AlertTriangle className="w-3 h-3 text-amber-500" />
                              <span className="text-xs text-slate-500">
                                {feat}: <span className="text-slate-700 font-medium">{Number(val).toFixed(3)}</span>
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-xs text-slate-400">—</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-xs text-slate-500 font-mono">{log.requester_id || '—'}</span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="px-6 py-16 text-center">
                    <ScrollText className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                    <p className="text-slate-500 font-medium">No audit logs yet</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Predictions will appear here after scoring employees via the Decision Cockpit.
                    </p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200">
            <p className="text-xs text-slate-500">
              Page {page} of {totalPages} ({data?.total_count} total records)
            </p>
            <div className="flex items-center gap-2">
              <button
                id="btn-prev-page"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="p-1.5 rounded-lg bg-white border border-slate-300 text-slate-600 hover:text-slate-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                id="btn-next-page"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="p-1.5 rounded-lg bg-white border border-slate-300 text-slate-600 hover:text-slate-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
}

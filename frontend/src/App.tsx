import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, DonutChart, BarChart } from '@tremor/react';
import { Users, Activity, DollarSign, LayoutDashboard, ShieldCheck, AlertCircle, ScrollText, TrendingUp, Radar, Lock, LogOut, Settings, type LucideIcon } from 'lucide-react';
import { motion } from 'framer-motion';
import DecisionCockpit from './components/DecisionCockpit';
import EmployeeProfiles from './components/EmployeeProfiles';
import RetentionStrategy from './components/RetentionStrategy';
import AuditTrail from './components/AuditTrail';
import ErrorBoundary from './components/ErrorBoundary';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import DriftMonitor from './components/DriftMonitor';
import LoginPage from './components/LoginPage';
import AdminDashboard from './components/AdminDashboard';
import { api, type DashboardSummary } from './lib/api';
import { useAuth, AccessGuard, Permission, type PermissionValue } from './lib/auth';

// ── Tab Configuration with RBAC ──────────────────────────────────────
interface TabConfig {
  name: string;
  icon: LucideIcon;
  /** Permission required to see this tab. If omitted, visible to all. */
  permission?: PermissionValue;
  /** Alternative: any of these permissions grants access */
  anyPermission?: PermissionValue[];
}

const ALL_TABS: TabConfig[] = [
  { name: 'Executive Overview',icon: LayoutDashboard, permission: Permission.DASHBOARD },
  { name: 'Predictive Analytics', icon: TrendingUp,      permission: Permission.DASHBOARD },
  { name: 'Employee Profiles', icon: Users,           permission: Permission.PREDICT },
  { name: 'Retention Strategy',icon: ShieldCheck,     permission: Permission.DASHBOARD },
  { name: 'Drift Monitor',     icon: Radar,           permission: Permission.SYSTEM },
  { name: 'Decision Cockpit',  icon: Activity,        permission: Permission.PREDICT },
  { name: 'Audit Trail',       icon: ScrollText,      permission: Permission.AUDIT },
  { name: 'Admin',             icon: Settings,        permission: Permission.SYSTEM },
];

// Fallback data when API is unavailable
const FALLBACK_DATA: DashboardSummary = {
  total_employees_scored: 0,
  high_risk_count: 0,
  medium_risk_count: 0,
  low_risk_count: 0,
  total_value_at_risk: 0,
  average_risk_probability: 0,
  top_systemic_drivers: [],
};

export default function App() {
  const { role, roleLabel, roleSubtitle, roleColor, displayName, can, isLoading: authLoading, isAuthenticated, logout } = useAuth();

  // Gate: show login page when not authenticated
  if (!isAuthenticated) {
    return <LoginPage />;
  }

  // Filter tabs based on user permissions
  const visibleTabs = ALL_TABS.filter((tab) => {
    if (tab.permission) return can(tab.permission);
    if (tab.anyPermission) return tab.anyPermission.some((p) => can(p));
    return true; // No permission required
  });

  const [activeTab, setActiveTab] = useState('');

  // Set initial active tab to first visible tab
  useEffect(() => {
    // Clean up any stale URL paths (like /cockpit) from previous hard navigation
    if (window.location.pathname !== '/') {
      window.history.replaceState({}, '', '/');
    }
    if (visibleTabs.length > 0 && !visibleTabs.find((t) => t.name === activeTab)) {
      setActiveTab(visibleTabs[0].name);
    }
  }, [visibleTabs, activeTab]);

  // Real API data for dashboard overview
  const { data: dashboardData, isLoading, isError, error } = useQuery({
    queryKey: ['dashboard-summary'],
    queryFn: api.getDashboardSummary,
    retry: 2,
    refetchInterval: 60000, // Refresh every 60 seconds
    staleTime: 30000,
    enabled: can(Permission.DASHBOARD),
  });

  // Health check
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: api.healthCheck,
    retry: 1,
    refetchInterval: 15000,
  });

  const summary = dashboardData || FALLBACK_DATA;
  const isModelOnline = health?.model_loaded ?? false;

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-background text-foreground overflow-hidden font-sans">
        
        {/* Sidebar */}
        <aside className="w-64 border-r border-navy-dark bg-navy text-white flex flex-col">
          <div className="p-6">
            <h1 className="text-xl font-bold bg-gradient-to-r from-teal-light to-emerald-400 bg-clip-text text-transparent">
              HR Attrition Intel
            </h1>
            <p className="text-xs text-slate-400 mt-1">AI-Powered Risk Engine v1.1</p>
          </div>
          
          <nav className="flex-1 px-4 space-y-2 mt-4">
            {authLoading ? (
              // Skeleton while auth is loading
              <>
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-11 bg-zinc-800/30 rounded-xl animate-pulse" />
                ))}
              </>
            ) : (
              visibleTabs.map((item) => (
                <button
                  key={item.name}
                  id={`nav-${item.name.toLowerCase().replace(/\s/g, '-')}`}
                  onClick={() => setActiveTab(item.name)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                    activeTab === item.name 
                      ? 'bg-white/10 text-teal-light border border-white/5 shadow-sm' 
                      : 'text-slate-300 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  <span className="text-sm font-medium">{item.name}</span>
                </button>
              ))
            )}
          </nav>

          {/* RBAC-Aware User Identity */}
          <div className="p-6 border-t border-zinc-800">
            <div className="flex items-center gap-3">
              <div className={`w-8 h-8 rounded-full bg-gradient-to-tr ${roleColor} flex items-center justify-center`}>
                <span className="text-xs font-bold text-white">{(displayName || roleLabel).charAt(0).toUpperCase()}</span>
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium truncate">{displayName || roleLabel}</p>
                <p className="text-xs text-zinc-500 truncate">{roleSubtitle}</p>
              </div>
            </div>
            {/* Role badge + Logout */}
            <div className="mt-3 flex items-center justify-between">
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium tracking-wider uppercase bg-zinc-800/80 border border-zinc-700/50 ${
                role === 'admin' ? 'text-emerald-400' :
                role === 'hr_partner' ? 'text-violet-400' :
                role === 'analyst' ? 'text-amber-400' :
                'text-zinc-400'
              }`}>
                <Lock className="w-2.5 h-2.5" />
                {role.replace('_', ' ')}
              </span>
              <button
                onClick={logout}
                title="Sign out"
                className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto bg-background">
          <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur-sm px-8 py-5 flex justify-between items-center shadow-sm">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight text-slate-900">{activeTab}</h2>
              <p className="text-sm text-slate-500 mt-1">Real-time systemic risk analysis.</p>
            </div>
            <div className="flex items-center gap-4">
              <span className="flex h-2 w-2 relative">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isModelOnline ? 'bg-emerald-400' : 'bg-red-400'}`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${isModelOnline ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
              </span>
              <span className="text-xs text-zinc-400 font-medium tracking-wider">
                {isModelOnline ? 'MODEL ONLINE' : 'MODEL OFFLINE'}
              </span>
            </div>
          </header>

          {activeTab === 'Executive Overview' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.DASHBOARD} fallback={<AccessDeniedPanel />}>
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="p-8 max-w-7xl mx-auto space-y-6"
                >
                  {/* API Error Banner */}
                  {isError && (
                    <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-amber-200">
                          Unable to load live data from the API
                        </p>
                        <p className="text-xs text-amber-400/70 mt-1">
                          {(error as any)?.detail || 'Backend may be offline. Showing cached data.'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* KPI Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
                      <div className="flex items-center gap-3 text-slate-500 mb-4">
                        <DollarSign className="w-5 h-5 text-rose-500" />
                        <p className="text-sm font-medium">Total Value at Risk</p>
                      </div>
                      {isLoading ? (
                        <div className="h-10 w-32 bg-slate-100 rounded animate-pulse" />
                      ) : (
                        <p className="text-4xl font-bold text-slate-900 tracking-tight flex items-baseline gap-2">
                          {(role === 'admin' || role === 'hr_partner') 
                            ? `$${(summary.total_value_at_risk / 1000000).toFixed(2)}M` 
                            : '***'}
                          {(role !== 'admin' && role !== 'hr_partner') && (
                            <span className="text-xs text-slate-400 font-normal">Masked</span>
                          )}
                        </p>
                      )}
                      <p className="text-xs text-slate-500 mt-2">
                        {summary.total_employees_scored} employees scored
                      </p>
                    </Card>

                    <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
                      <div className="flex items-center gap-3 text-slate-500 mb-4">
                        <Users className="w-5 h-5 text-amber-500" />
                        <p className="text-sm font-medium">High Risk Employees</p>
                      </div>
                      {isLoading ? (
                        <div className="h-10 w-16 bg-slate-100 rounded animate-pulse" />
                      ) : (
                        <p className="text-4xl font-bold text-slate-900 tracking-tight">
                          {summary.high_risk_count}
                        </p>
                      )}
                      <p className="text-xs text-slate-500 mt-2">Requires immediate action</p>
                    </Card>

                    <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
                      <div className="flex items-center gap-3 text-slate-500 mb-4">
                        <Activity className="w-5 h-5 text-emerald-500" />
                        <p className="text-sm font-medium">Average Fleet Risk</p>
                      </div>
                      {isLoading ? (
                        <div className="h-10 w-20 bg-slate-100 rounded animate-pulse" />
                      ) : (
                        <p className="text-4xl font-bold text-slate-900 tracking-tight">
                          {(summary.average_risk_probability * 100).toFixed(1)}%
                        </p>
                      )}
                      <p className="text-xs text-slate-500 mt-2">Across all departments</p>
                    </Card>
                  </div>

                  {/* Charts Area */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
                    <Card className="col-span-2 bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
                      <h3 className="text-lg font-medium text-slate-900 mb-6">Top Systemic Risk Drivers (SHAP)</h3>
                      {summary.top_systemic_drivers.length > 0 ? (
                        <BarChart
                          className="h-72 mt-4"
                          data={summary.top_systemic_drivers.slice(0, 5).map((d: any) => ({
                            name: String(d.Risk_Tier || d.feature || 'Unknown'),
                            impact: Number(d.Total_Expected_Loss || d.mean_abs_shap || 0),
                          }))}
                          index="name"
                          categories={["impact"]}
                          colors={["rose"]}
                          valueFormatter={(number) => `$${Intl.NumberFormat('us', { notation: 'compact' }).format(number)}`}
                          yAxisWidth={65}
                          showAnimation={true}
                        />
                      ) : (
                        <div className="h-72 flex items-center justify-center text-zinc-500 text-sm">
                          Run the training pipeline to generate risk data
                        </div>
                      )}
                    </Card>

                    <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6 flex flex-col justify-between">
                      <h3 className="text-lg font-medium text-slate-900 mb-6">Risk Distribution</h3>
                      <div className="flex-1 flex items-center justify-center">
                        <DonutChart
                          className="h-48"
                          data={[
                            { name: 'High Risk', count: summary.high_risk_count },
                            { name: 'Medium Risk', count: summary.medium_risk_count },
                            { name: 'Low Risk', count: summary.low_risk_count },
                          ]}
                          category="count"
                          index="name"
                          colors={["rose", "amber", "emerald"]}
                          showAnimation={true}
                        />
                      </div>
                    </Card>
                  </div>
                </motion.div>
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Predictive Analytics' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.DASHBOARD} fallback={<AccessDeniedPanel />}>
                <AnalyticsDashboard />
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Decision Cockpit' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.PREDICT} fallback={<AccessDeniedPanel />}>
                <DecisionCockpit />
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Employee Profiles' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.PREDICT} fallback={<AccessDeniedPanel />}>
                <EmployeeProfiles />
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Retention Strategy' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.DASHBOARD} fallback={<AccessDeniedPanel />}>
                <RetentionStrategy />
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Drift Monitor' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.SYSTEM} fallback={<AccessDeniedPanel />}>
                <DriftMonitor />
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Audit Trail' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.AUDIT} fallback={<AccessDeniedPanel />}>
                <AuditTrail />
              </AccessGuard>
            </ErrorBoundary>
          )}

          {activeTab === 'Admin' && (
            <ErrorBoundary>
              <AccessGuard permission={Permission.SYSTEM} fallback={<AccessDeniedPanel />}>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="p-8 max-w-6xl mx-auto"
                >
                  <AdminDashboard />
                </motion.div>
              </AccessGuard>
            </ErrorBoundary>
          )}
        </main>
      </div>
    </ErrorBoundary>
  );
}


// ── Access Denied Panel ──────────────────────────────────────────────
function AccessDeniedPanel() {
  const { role, roleLabel } = useAuth();
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="p-8 max-w-lg mx-auto mt-20"
    >
      <div className="bg-white border border-slate-200 shadow-tonal rounded-2xl p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-5 rounded-2xl bg-red-50 border border-red-100 flex items-center justify-center">
          <Lock className="w-7 h-7 text-red-500" />
        </div>
        <h3 className="text-lg font-semibold text-slate-900 mb-2">Access Restricted</h3>
        <p className="text-sm text-slate-500 leading-relaxed">
          Your role <span className="font-mono text-amber-600">({role})</span> does not have
          permission to access this section.
        </p>
        <p className="text-xs text-slate-400 mt-3">
          Logged in as <strong className="text-slate-700">{roleLabel}</strong>.
          Contact your system administrator for access.
        </p>
      </div>
    </motion.div>
  );
}

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Card, 
  Table, 
  TableHead, 
  TableRow, 
  TableHeaderCell, 
  TableBody, 
  TableCell,
  Badge,
  BarChart,
  Select,
  SelectItem,
  TextInput,
  Grid
} from '@tremor/react';
import { motion } from 'framer-motion';
import { Search, AlertCircle, Building2, Briefcase, ShieldAlert, BrainCircuit, ChevronDown, ChevronUp } from 'lucide-react';
import { api } from '../lib/api';
import { useAuth } from '../lib/auth';

function ExpandedEmployeeRow({ emp }: { emp: any }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['employee-narrative', emp.EmployeeNumber],
    queryFn: () => api.getEmployeeNarrative(emp.EmployeeNumber),
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="p-8 text-center text-slate-500 animate-pulse">
        <BrainCircuit className="w-6 h-6 text-teal-500 mx-auto mb-2 animate-spin-slow" />
        Generating dynamic AI insights via LLM...
      </div>
    );
  }
  if (isError || !data) {
    return <div className="p-8 text-center text-rose-500">Failed to load AI narrative.</div>;
  }

  const interventions = Array.isArray(data.Recommended_Interventions) 
    ? data.Recommended_Interventions 
    : typeof data.Recommended_Interventions === 'string'
      ? data.Recommended_Interventions.split('\n').filter((i: string) => i.trim().length > 0)
      : [];

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6"
    >
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-teal-600" />
          Contextual Narrative (SHAP + LLM Explainability)
        </h4>
        <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap break-words">
          {data.AI_Contextual_Narrative}
        </p>
        {data.Top_Risk_Drivers && data.Top_Risk_Drivers.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-2">
            {data.Top_Risk_Drivers.slice(0, 3).map((driver: any, idx: number) => (
              <span key={idx} className="inline-flex items-center px-2 py-1 rounded bg-rose-100 text-rose-700 text-xs font-medium border border-rose-200">
                {driver.Feature || driver}
              </span>
            ))}
          </div>
        )}
      </div>
      
      <div className="space-y-3 pl-0 md:pl-6 md:border-l border-slate-200">
        <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-amber-500" />
          Recommended Interventions
        </h4>
        <ul className="text-sm text-slate-600 space-y-2 list-disc pl-4">
          {interventions.length > 0 ? interventions.map((item: string, idx: number) => (
            <li key={idx}>{item.replace(/^- /, '')}</li>
          )) : (
            <li>No specific interventions generated.</li>
          )}
        </ul>
        <div className="pt-3">
          <button 
            onClick={() => {
              localStorage.setItem('simulatorEmployeeId', emp.EmployeeNumber.toString());
              document.getElementById('nav-decision-cockpit')?.click();
            }} 
            className="text-teal-600 hover:text-teal-700 text-sm font-medium transition-colors cursor-pointer"
          >
            Open in What-If Simulator &rarr;
          </button>
        </div>
      </div>
    </motion.div>
  );
}

export default function AnalyticsDashboard() {
  const { role } = useAuth();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDepartment, setSelectedDepartment] = useState<string>('all');
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 50;

  // Reset page when filters change
  React.useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, selectedDepartment, selectedTier]);

  const { data: employees = [], isLoading, isError } = useQuery({
    queryKey: ['dashboard-employees'],
    queryFn: api.getDashboardEmployees,
    staleTime: 60000,
  });

  // Derived metrics and filters
  const filteredEmployees = useMemo(() => {
    return employees.filter((emp) => {
      const empNumStr = emp?.EmployeeNumber?.toString() || '';
      const empRole = emp?.JobRole?.toLowerCase() || '';
      const matchesSearch = empNumStr.includes(searchTerm) || empRole.includes(searchTerm.toLowerCase());
      const matchesDept = selectedDepartment === 'all' || emp.Department === selectedDepartment;
      const matchesTier = selectedTier === 'all' || emp.Risk_Tier === selectedTier;
      return matchesSearch && matchesDept && matchesTier;
    }).sort((a, b) => (b?.Predicted_Probability || 0) - (a?.Predicted_Probability || 0)); // Sort by highest risk
  }, [employees, searchTerm, selectedDepartment, selectedTier]);

  const totalPages = Math.ceil(filteredEmployees.length / rowsPerPage);
  const paginatedEmployees = filteredEmployees.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

  const departments = useMemo(() => Array.from(new Set(employees.map(e => e.Department))).filter(Boolean), [employees]);

  const departmentLossData = useMemo(() => {
    const deptMap: Record<string, number> = {};
    employees.forEach(e => {
      if (e.Department && e.Expected_Loss) {
        deptMap[e.Department] = (deptMap[e.Department] || 0) + e.Expected_Loss;
      }
    });
    return Object.entries(deptMap)
      .map(([name, loss]) => ({ name, 'Expected Loss ($)': loss }))
      .sort((a, b) => b['Expected Loss ($)'] - a['Expected Loss ($)']);
  }, [employees]);

  const departmentRiskData = useMemo(() => {
    const deptMap: Record<string, { high: number, medium: number, low: number }> = {};
    employees.forEach(e => {
      if (e.Department) {
        if (!deptMap[e.Department]) deptMap[e.Department] = { high: 0, medium: 0, low: 0 };
        if (e.Risk_Tier === 'High') deptMap[e.Department].high++;
        else if (e.Risk_Tier === 'Medium') deptMap[e.Department].medium++;
        else deptMap[e.Department].low++;
      }
    });
    return Object.entries(deptMap).map(([name, counts]) => ({
      name,
      ...counts,
      total: counts.high + counts.medium + counts.low
    })).sort((a, b) => b.high - a.high); // Sort by highest risk cases
  }, [employees]);

  if (isLoading) {
    return (
      <div className="p-8 max-w-7xl mx-auto flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-4 text-zinc-500">
          <div className="w-8 h-8 rounded-full border-2 border-emerald-500 border-t-transparent animate-spin" />
          <p>Loading analytics data...</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <p className="text-sm font-medium text-rose-200">
            Failed to load analytics data. Ensure the training pipeline has been run.
          </p>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="p-8 max-w-7xl mx-auto space-y-8"
    >
      <div className="flex flex-col gap-2">
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">Fleet-Level Analytics</h2>
        <p className="text-slate-500 text-sm">Deep observability into systemic risk and financial exposure.</p>
      </div>

      {/* Fleet-Level Analytics Grid */}
      <Grid numItems={1} numItemsMd={2} className="gap-6">
        {/* Financial Exposure by Department */}
        <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <Building2 className="w-5 h-5 text-emerald-500" />
            <h3 className="text-lg font-medium text-slate-900">Financial Exposure</h3>
            {(role !== 'admin' && role !== 'hr_partner') && (
              <span className="ml-auto flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] bg-slate-100 text-slate-500 border border-slate-200 label-caps">
                <ShieldAlert className="w-3 h-3" />
                Masked
              </span>
            )}
          </div>
          
          {(role === 'admin' || role === 'hr_partner') ? (
            <BarChart
              className="h-64 mt-4 data-mono"
              data={departmentLossData}
              index="name"
              categories={["Expected Loss ($)"]}
              colors={["emerald"]}
              valueFormatter={(number) => `$${Intl.NumberFormat('us', { notation: 'compact' }).format(number)}`}
              yAxisWidth={60}
              showAnimation={true}
            />
          ) : (
            <div className="h-64 mt-4 flex flex-col items-center justify-center border border-dashed border-slate-200 rounded-xl bg-slate-50">
              <ShieldAlert className="w-8 h-8 text-slate-400 mb-3" />
              <p className="text-slate-600 font-medium">Financial Data Masked</p>
              <p className="text-slate-500 text-sm mt-1 text-center max-w-xs">Your role does not have permission to view departmental financial exposure.</p>
            </div>
          )}
        </Card>

        {/* AI-Powered Risk Heatmap */}
        <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <BrainCircuit className="w-5 h-5 text-rose-500" />
            <h3 className="text-lg font-medium text-slate-900">Risk Distribution Heatmap</h3>
          </div>
          <div className="flex flex-col justify-center h-64 overflow-y-auto pr-2 space-y-5">
            {departmentRiskData.length === 0 ? (
               <p className="text-slate-500 text-sm text-center">No departmental risk data available.</p>
            ) : (
              departmentRiskData.map(dept => (
                <div key={dept.name} className="flex flex-col gap-1.5">
                  <div className="flex justify-between items-end">
                    <span className="text-sm font-medium text-slate-700">{dept.name}</span>
                    <span className="text-slate-500 label-caps">{dept.total} Emp</span>
                  </div>
                  <div className="h-2 flex w-full rounded-full overflow-hidden bg-slate-100">
                    <div className="bg-rose-500 transition-all duration-500" style={{ width: `${(dept.high / dept.total) * 100}%` }} title={`High: ${dept.high}`} />
                    <div className="bg-amber-400 transition-all duration-500" style={{ width: `${(dept.medium / dept.total) * 100}%` }} title={`Medium: ${dept.medium}`} />
                    <div className="bg-emerald-400 transition-all duration-500" style={{ width: `${(dept.low / dept.total) * 100}%` }} title={`Low: ${dept.low}`} />
                  </div>
                  <div className="flex gap-3 text-[10px] text-slate-400 data-mono">
                    <span className="text-rose-600">{dept.high} High</span>
                    <span className="text-amber-600">{dept.medium} Med</span>
                    <span className="text-emerald-600">{dept.low} Low</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </Grid>

      {/* Employee Risk Table */}
      <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-0 overflow-hidden flex flex-col">
        <div className="p-6 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Briefcase className="w-5 h-5 text-teal-500" />
            <h3 className="text-lg font-medium text-slate-900">Employee Risk Directory</h3>
          </div>
          
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <div className="relative w-full sm:w-64">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
              <TextInput
                placeholder="Search Emp ID or Role..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9 bg-white border-slate-200 text-slate-900 shadow-sm"
              />
            </div>
            <Select 
              value={selectedDepartment} 
              onValueChange={setSelectedDepartment}
              className="w-40"
            >
              <SelectItem value="all">All Departments</SelectItem>
              {departments.map(d => (
                <SelectItem key={d} value={d}>{d}</SelectItem>
              ))}
            </Select>
            <Select 
              value={selectedTier} 
              onValueChange={setSelectedTier}
              className="w-32"
            >
              <SelectItem value="all">All Tiers</SelectItem>
              <SelectItem value="High">High Risk</SelectItem>
              <SelectItem value="Medium">Medium Risk</SelectItem>
              <SelectItem value="Low">Low Risk</SelectItem>
            </Select>
          </div>
        </div>

        <div className="overflow-x-auto max-h-[600px]">
          <Table>
            <TableHead className="bg-slate-50 sticky top-0 z-10 border-b border-slate-200">
              <TableRow>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent">Emp ID</TableHeaderCell>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent">Department</TableHeaderCell>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent">Job Role</TableHeaderCell>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent">Risk Prob</TableHeaderCell>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent">Tier</TableHeaderCell>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent text-right">Rep Cost</TableHeaderCell>
                <TableHeaderCell className="text-slate-600 label-caps bg-transparent text-right">Exp Loss</TableHeaderCell>
                <TableHeaderCell className="bg-transparent"></TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredEmployees.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-slate-500">
                    No employees match the current filters.
                  </TableCell>
                </TableRow>
              ) : (
                paginatedEmployees.map((emp) => (
                  <React.Fragment key={emp.EmployeeNumber}>
                    <TableRow 
                      className="hover:bg-slate-50/50 transition-colors border-b border-slate-100 cursor-pointer"
                      onClick={() => setExpandedRow(expandedRow === emp.EmployeeNumber ? null : emp.EmployeeNumber)}
                    >
                      <TableCell className="text-slate-900 data-mono font-bold">#{emp.EmployeeNumber}</TableCell>
                      <TableCell className="text-slate-600 text-sm">{emp.Department}</TableCell>
                      <TableCell className="text-slate-600 text-sm">{emp.JobRole}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="text-slate-700 data-mono">{(emp.Predicted_Probability * 100).toFixed(1)}%</span>
                          <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                emp.Risk_Tier === 'High' ? 'bg-rose-500' : 
                                emp.Risk_Tier === 'Medium' ? 'bg-amber-400' : 'bg-emerald-400'
                              }`}
                              style={{ width: `${emp.Predicted_Probability * 100}%` }}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge 
                          color={
                            emp.Risk_Tier === 'High' ? 'rose' : 
                            emp.Risk_Tier === 'Medium' ? 'amber' : 'emerald'
                          }
                          className="bg-opacity-10 border border-slate-200/50"
                        >
                          {emp.Risk_Tier}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right text-slate-600 data-mono">
                        {(role === 'admin' || role === 'hr_partner') ? (
                          `$${Intl.NumberFormat('us').format(Math.round(emp.Replacement_Cost))}`
                        ) : (
                          <span className="text-slate-400">***</span>
                        )}
                      </TableCell>
                      <TableCell className="text-right font-medium text-slate-900 data-mono">
                        {(role === 'admin' || role === 'hr_partner') ? (
                          `$${Intl.NumberFormat('us').format(Math.round(emp.Expected_Loss))}`
                        ) : (
                          <span className="text-slate-400">***</span>
                        )}
                      </TableCell>
                      <TableCell className="text-slate-400 text-right">
                        {expandedRow === emp.EmployeeNumber ? <ChevronUp className="w-4 h-4 inline" /> : <ChevronDown className="w-4 h-4 inline" />}
                      </TableCell>
                    </TableRow>
                    
                    {/* Expandable SHAP Narrative & Intervention */}
                    {expandedRow === emp.EmployeeNumber && (
                      <TableRow className="bg-slate-50 border-b border-slate-200">
                        <TableCell colSpan={8} className="p-0 whitespace-normal">
                          <ExpandedEmployeeRow emp={emp} />
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                ))
              )}
            </TableBody>
          </Table>
        </div>
        <div className="p-4 border-t border-slate-200 bg-slate-50 flex justify-between items-center text-sm text-slate-500">
          <div className="flex items-center gap-4">
            <span>Showing {paginatedEmployees.length} of {filteredEmployees.length} employees</span>
            {totalPages > 1 && (
              <div className="flex gap-2">
                <button 
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-2 py-1 rounded bg-white hover:bg-slate-50 border border-slate-300 disabled:opacity-50 transition-colors"
                >Prev</button>
                <span className="px-2 py-1 font-medium">Page {currentPage} of {totalPages}</span>
                <button 
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="px-2 py-1 rounded bg-white hover:bg-slate-50 border border-slate-300 disabled:opacity-50 transition-colors"
                >Next</button>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            {(role !== 'admin' && role !== 'hr_partner') && (
              <span className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] bg-slate-200 text-slate-600 border border-slate-300">
                <ShieldAlert className="w-3 h-3" />
                Financial data masked
              </span>
            )}
            <span>Total Expected Loss in View: <strong className="text-slate-900 font-semibold">
              {(role === 'admin' || role === 'hr_partner') 
                ? `$${Intl.NumberFormat('us').format(Math.round(filteredEmployees.reduce((sum, e) => sum + e.Expected_Loss, 0)))}`
                : '***'}
            </strong></span>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

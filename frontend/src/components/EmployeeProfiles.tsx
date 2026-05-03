import { useState } from 'react';
import { Card, TextInput, Select, SelectItem, DonutChart, Metric, ProgressBar } from '@tremor/react';
import { motion } from 'framer-motion';
import { Search, AlertTriangle } from 'lucide-react';
import { api } from '../lib/api';
import { useQuery } from '@tanstack/react-query';

export default function EmployeeProfiles() {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('all');

  const { data: employees = [], isLoading } = useQuery({
    queryKey: ['dashboardEmployees'],
    queryFn: api.getDashboardEmployees
  });

  // Map backend EmployeeRiskScore to the profile format
  const profiles = employees.map(emp => ({
    id: emp.EmployeeNumber,
    name: `Employee ${emp.EmployeeNumber}`, // Masked for privacy
    role: emp.JobRole,
    dept: emp.Department,
    tenure: emp.YearsAtCompany ? `${emp.YearsAtCompany} Years` : 'Data unavailable',
    location: 'Corporate HQ',
    tier: emp.PerformanceRating || 0, 
    risk: Math.round(emp.Predicted_Probability * 100),
    salary: emp.Annual_Salary || 0,
    growth: emp.PercentSalaryHike || 0,
    expectedLoss: emp.Expected_Loss || 0,
    satisfaction: {
      job: emp.JobSatisfaction || 0,
      env: emp.EnvironmentSatisfaction || 0,
      rel: emp.RelationshipSatisfaction || 0,
      wlb: emp.WorkLifeBalance || 0
    }
  }));

  const filteredProfiles = profiles.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase()) || p.role.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesDept = departmentFilter === 'all' || p.dept === departmentFilter;
    return matchesSearch && matchesDept;
  });

  const selectedProfile = profiles.find(p => p.id === selectedId) || profiles[0] || null;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6 font-inter">
      {/* Search and Filter Top Bar */}
      <div className="flex gap-4 mb-6">
        <div className="relative flex-grow max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-4 h-4" />
          <TextInput
            icon={Search}
            placeholder="Search talent database..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <Select value={departmentFilter} onValueChange={setDepartmentFilter} placeholder="Filter by Department">
          <SelectItem value="all">All Departments</SelectItem>
          <SelectItem value="Research & Development">Research & Development</SelectItem>
          <SelectItem value="Sales">Sales</SelectItem>
          <SelectItem value="Human Resources">Human Resources</SelectItem>
        </Select>
      </div>

      {isLoading && (
        <div className="flex justify-center p-12"><div className="w-8 h-8 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div></div>
      )}

      {!isLoading && !selectedProfile && (
        <div className="text-center p-12 text-slate-500">No employees found.</div>
      )}

      {!isLoading && selectedProfile && (
        <div className="grid grid-cols-12 gap-6">
          {/* Sidebar list for selecting profiles */}
          <div className="col-span-12 lg:col-span-3">
            <Card className="h-[600px] overflow-y-auto p-0">
              <div className="p-4 border-b border-slate-100 bg-slate-50 sticky top-0 font-semibold text-sm">
                Employee Directory
              </div>
              <div className="divide-y divide-slate-100">
                {filteredProfiles.map(p => (
                  <button 
                    key={p.id}
                    onClick={() => setSelectedId(p.id)}
                    className={`w-full text-left p-4 hover:bg-slate-50 transition-colors ${selectedId === p.id || (!selectedId && selectedProfile.id === p.id) ? 'bg-teal-50 border-l-4 border-teal-500' : ''}`}
                  >
                    <div className="font-semibold text-sm text-slate-900">{p.name}</div>
                    <div className="text-xs text-slate-500 mt-1">{p.role}</div>
                    <div className="mt-2">
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider ${p.risk >= 60 ? 'bg-rose-100 text-rose-700' : p.risk >= 30 ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'}`}>
                        {p.risk}% Risk
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </Card>
          </div>

          <div className="col-span-12 lg:col-span-9 space-y-6">
            {/* Hero Profile Section */}
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
              <Card className="flex items-center gap-8 bg-white border-slate-200 shadow-sm p-6 rounded-xl">
          <div className="w-32 h-32 rounded-xl overflow-hidden ring-4 ring-slate-50 flex-shrink-0 bg-slate-200">
            <img src={`https://api.dicebear.com/7.x/notionists/svg?seed=${selectedProfile.name}`} alt="Profile" className="w-full h-full object-cover" />
          </div>
          <div className="flex-grow">
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-2xl font-bold text-slate-900">{selectedProfile.name}</h1>
              <span className="bg-teal-50 text-teal-700 px-3 py-1 rounded-full text-[10px] font-bold tracking-widest border border-teal-200 uppercase">Active Talent</span>
            </div>
            <p className="text-lg text-slate-500 mb-4">{selectedProfile.role} • {selectedProfile.dept}</p>
            
            <div className="flex gap-12">
              <div>
                <span className="block text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-widest">Tenure</span>
                <span className="text-sm font-semibold">{selectedProfile.tenure}</span>
              </div>
              <div>
                <span className="block text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-widest">Location</span>
                <span className="text-sm font-semibold">{selectedProfile.location}</span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <span className="block text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-widest">Performance Tier</span>
            <div className="text-4xl font-black text-teal-600">{selectedProfile.tier.toFixed(1)}<span className="text-lg text-slate-300 ml-1">/ 5.0</span></div>
          </div>
        </Card>
      </motion.div>

      {/* Deep Dive Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Risk Assessment */}
        <div className="col-span-12 lg:col-span-4">
          <Card className={`h-full ${selectedProfile.risk >= 60 ? 'bg-rose-50/50 border-rose-100' : selectedProfile.risk >= 30 ? 'bg-amber-50/50 border-amber-100' : 'bg-emerald-50/50 border-emerald-100'}`}>
            <div className="flex items-center justify-between mb-6">
              <h3 className={`text-lg font-semibold flex items-center gap-2 ${selectedProfile.risk >= 60 ? 'text-rose-700' : selectedProfile.risk >= 30 ? 'text-amber-700' : 'text-emerald-700'}`}>
                <AlertTriangle className="w-5 h-5" /> Attrition Risk
              </h3>
              <span className={`text-xs font-bold text-white px-2 py-1 rounded uppercase ${selectedProfile.risk >= 60 ? 'bg-rose-500' : selectedProfile.risk >= 30 ? 'bg-amber-500' : 'bg-emerald-500'}`}>
                {selectedProfile.risk >= 60 ? 'High' : selectedProfile.risk >= 30 ? 'Medium' : 'Low'} Risk
              </span>
            </div>
            
            <div className="mb-6">
              <Metric className={`mb-2 ${selectedProfile.risk >= 60 ? 'text-rose-600' : selectedProfile.risk >= 30 ? 'text-amber-600' : 'text-emerald-600'}`}>
                {selectedProfile.risk}%
              </Metric>
              <ProgressBar 
                value={selectedProfile.risk} 
                color={selectedProfile.risk >= 60 ? 'rose' : selectedProfile.risk >= 30 ? 'amber' : 'emerald'} 
                className="mt-2" 
              />
            </div>

            <p className="text-xs font-bold text-rose-800 mb-2 uppercase tracking-widest">Financial Impact</p>
            <ul className="space-y-3 mb-6">
              <li className="flex items-start gap-2 text-sm text-slate-700">
                <AlertTriangle className="w-4 h-4 mt-0.5 text-rose-500" />
                Expected Loss: <strong>${Math.round(selectedProfile.expectedLoss).toLocaleString()}</strong>
              </li>
              <li className="flex items-start gap-2 text-sm text-slate-700">
                <AlertTriangle className="w-4 h-4 mt-0.5 text-rose-500" />
                Annual Salary Base: <strong>${Math.round(selectedProfile.salary).toLocaleString()}</strong>
              </li>
            </ul>

            <div className="pt-4 border-t border-rose-200">
              <p className="text-xs font-bold text-rose-800 mb-3 uppercase tracking-widest">AI Retention Actions</p>
              <div className="space-y-2">
                {selectedProfile.satisfaction.wlb < 3 && (
                  <button className="w-full text-left bg-white border border-rose-200 p-3 rounded-lg hover:bg-rose-50 transition-colors text-sm font-semibold text-slate-700">
                    Propose Flexible/Remote Work Plan
                  </button>
                )}
                {selectedProfile.growth < 15 && (
                  <button className="w-full text-left bg-white border border-rose-200 p-3 rounded-lg hover:bg-rose-50 transition-colors text-sm font-semibold text-slate-700">
                    Review Compensation Model
                  </button>
                )}
                <button className="w-full text-left bg-white border border-rose-200 p-3 rounded-lg hover:bg-rose-50 transition-colors text-sm font-semibold text-slate-700">
                  Schedule 1:1 Intervention
                </button>
              </div>
            </div>
          </Card>
        </div>

        {/* Satisfaction Matrix */}
        <div className="col-span-12 lg:col-span-8">
          <Card className="h-full">
            <h3 className="text-lg font-semibold text-slate-900 mb-6">Satisfaction Matrix</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              {[
                { label: 'Job Satisfaction', val: selectedProfile.satisfaction.job, color: selectedProfile.satisfaction.job >= 3 ? 'emerald' : selectedProfile.satisfaction.job == 2 ? 'amber' : 'rose' },
                { label: 'Environment', val: selectedProfile.satisfaction.env, color: selectedProfile.satisfaction.env >= 3 ? 'emerald' : selectedProfile.satisfaction.env == 2 ? 'amber' : 'rose' },
                { label: 'Relationship', val: selectedProfile.satisfaction.rel, color: selectedProfile.satisfaction.rel >= 3 ? 'emerald' : selectedProfile.satisfaction.rel == 2 ? 'amber' : 'rose' },
                { label: 'Work-Life Balance', val: selectedProfile.satisfaction.wlb, color: selectedProfile.satisfaction.wlb >= 3 ? 'emerald' : selectedProfile.satisfaction.wlb == 2 ? 'amber' : 'rose' },
              ].map((item) => (
                <div key={item.label} className="text-center">
                  <DonutChart
                    className="h-24 w-24 mx-auto mb-4"
                    data={[{ name: 'Score', value: item.val }, { name: 'Empty', value: Math.max(0, 4 - item.val) }]}
                    category="value"
                    index="name"
                    colors={[item.color, "slate"]}
                    showTooltip={false}
                    showLabel={true}
                    valueFormatter={() => item.val.toString()}
                  />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{item.label}</span>
                </div>
              ))}
            </div>

            <div className="mt-8 pt-8 border-t border-slate-100 flex justify-between">
              <div>
                <span className="block text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-widest">Total Salary Growth</span>
                <span className="text-2xl font-bold text-teal-600">+{selectedProfile.growth}%</span>
              </div>
              <div>
                <span className="block text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-widest">Current Base Salary</span>
                <span className="text-2xl font-bold text-slate-900">${Math.round(selectedProfile.salary).toLocaleString()}</span>
              </div>
              <div>
                <span className="block text-[10px] font-bold text-slate-400 mb-1 uppercase tracking-widest">Market Pos.</span>
                <span className={`text-2xl font-bold ${selectedProfile.growth < 15 ? 'text-rose-500' : 'text-emerald-500'}`}>
                  {selectedProfile.growth < 15 ? 'Low' : 'Avg'} ({selectedProfile.growth}%)
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>
          </div>
        </div>
      )}
    </div>
  );
}

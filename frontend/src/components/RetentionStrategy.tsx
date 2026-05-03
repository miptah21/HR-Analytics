import { Card, TextInput, Select, SelectItem, BarChart, Badge } from '@tremor/react';
import { TrendingUp } from 'lucide-react';

export default function RetentionStrategy() {
  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 font-inter">
      {/* Header */}
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Retention Strategy Center</h2>
          <p className="text-sm text-slate-500 mt-1">Institutional intervention planning and ROI measurement dashboard.</p>
        </div>
        <div className="flex gap-6">
          <div className="flex flex-col items-end">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Global Retention Rate</span>
            <span className="text-4xl font-black text-slate-900">94.2%</span>
          </div>
          <div className="w-[1px] bg-slate-200"></div>
          <div className="flex flex-col items-end">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">YTD Impact</span>
            <span className="text-4xl font-black text-teal-600">+2.8%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Retention Impact Chart */}
        <div className="col-span-12 lg:col-span-8">
          <Card className="h-full">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-semibold text-slate-900">Retention Impact Analysis</h3>
            </div>
            
            <BarChart
              className="h-72 mt-4"
              data={[
                { category: 'Engagement', 'Pre-Program': 40, 'Post-Program': 85 },
                { category: 'Benefits', 'Pre-Program': 55, 'Post-Program': 75 },
                { category: 'Mentorship', 'Pre-Program': 80, 'Post-Program': 92 },
                { category: 'Compensation', 'Pre-Program': 40, 'Post-Program': 65 },
              ]}
              index="category"
              categories={['Pre-Program', 'Post-Program']}
              colors={['slate', 'teal']}
              valueFormatter={(val) => `${val}%`}
              yAxisWidth={48}
            />
          </Card>
        </div>

        {/* ROI Quick Card */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <Card className="bg-slate-900 text-white relative overflow-hidden h-40">
            <div className="relative z-10">
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Total ROI Estimate</h3>
              <div className="text-5xl font-black mb-2">$2.4M</div>
              <p className="text-xs text-slate-400">Cumulative savings from reduced replacement costs.</p>
            </div>
            <TrendingUp className="absolute -bottom-4 -right-4 w-32 h-32 text-slate-800 opacity-50" />
          </Card>

          <Card>
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">Program Coverage</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span>High-Risk Talent</span>
                  <span>88%</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div className="bg-teal-600 h-full w-[88%]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span>Leadership Track</span>
                  <span>64%</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div className="bg-emerald-500 h-full w-[64%]"></div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Active Interventions Table */}
        <div className="col-span-12 lg:col-span-8">
          <Card className="h-full">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Active Interventions</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Program Name</th>
                    <th className="py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Audience</th>
                    <th className="py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest text-right">Est. ROI</th>
                    <th className="py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  <tr className="hover:bg-slate-50">
                    <td className="py-4 font-semibold text-slate-900">Mentorship Program</td>
                    <td className="py-4 text-sm text-slate-600">Engineering L1-L2</td>
                    <td className="py-4 text-right font-mono text-sm font-semibold">$450k</td>
                    <td className="py-4 text-center">
                      <Badge color="teal">Ongoing</Badge>
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="py-4 font-semibold text-slate-900">Compensation Review</td>
                    <td className="py-4 text-sm text-slate-600">Fintech Specialists</td>
                    <td className="py-4 text-right font-mono text-sm font-semibold">$1.2M</td>
                    <td className="py-4 text-center">
                      <Badge color="amber">Planning</Badge>
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="py-4 font-semibold text-slate-900">Sabbatical Policy</td>
                    <td className="py-4 text-sm text-slate-600">Tenured (5y+)</td>
                    <td className="py-4 text-right font-mono text-sm font-semibold">$120k</td>
                    <td className="py-4 text-center">
                      <Badge color="slate">Completed</Badge>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Strategy Planner Form */}
        <div className="col-span-12 lg:col-span-4">
          <Card className="h-full bg-slate-50 border-dashed border-2 border-slate-200">
            <h3 className="text-lg font-semibold text-slate-900 mb-1">Strategy Planner</h3>
            <p className="text-xs text-slate-500 mb-6">Initialize a new retention intervention.</p>
            
            <form className="space-y-4">
              <div>
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Initiative Name</label>
                <TextInput placeholder="e.g. Wellness Credit Expansion" />
              </div>
              <div>
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Target Audience</label>
                <Select placeholder="Select segment...">
                  <SelectItem value="high">High-Risk High-Performers</SelectItem>
                  <SelectItem value="eng">Engineering Level 1-2</SelectItem>
                </Select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Budget</label>
                  <TextInput placeholder="$0.00" />
                </div>
                <div>
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Timeframe</label>
                  <TextInput placeholder="Q3-Q4" />
                </div>
              </div>
              <button 
                type="button"
                className="w-full py-2.5 bg-slate-900 text-white rounded-lg text-sm font-semibold hover:bg-slate-800 transition-colors mt-4"
              >
                DEPLOY STRATEGY
              </button>
            </form>
          </Card>
        </div>
      </div>
    </div>
  );
}

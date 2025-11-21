import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Shield, CheckCircle, AlertTriangle, Users, Eye, Clock, Zap, Globe } from 'lucide-react';

// Mock ZK-SNARK/STARK implementation for demonstration
class ZKPrivacyLayer {
  static generateZKProof(constitutionalProof, sensitiveData) {
    // In production: use circom, libsnark, or stark-js
    const proofData = {
      public_inputs: {
        constitutional_validity: constitutionalProof.constitutional_validity,
        jurisdiction: constitutionalProof.jurisdiction,
        threat_level: constitutionalProof.threat_assessment?.urgency || 0,
        timestamp: constitutionalProof.proof_timestamp
      },
      private_inputs: {
        // These remain hidden from public verification
        specific_sensors: sensitiveData.sensor_details,
        individual_votes: sensitiveData.voting_details,
        personal_data: sensitiveData.affected_individuals
      },
      zk_proof: `snark_${Math.random().toString(36).substring(2, 15)}`,
      verification_key: `vk_${Math.random().toString(36).substring(2, 10)}`
    };
    
    return proofData;
  }
  
  static verifyZKProof(zkProof) {
    // In production: actual zk-SNARK verification
    return {
      valid: true,
      verification_time_ms: Math.random() * 50 + 10,
      proof_size_bytes: 288, // Typical SNARK proof size
      public_inputs_verified: Object.keys(zkProof.public_inputs).length,
      privacy_preserved: true
    };
  }
}

// Mock governance data generator
const generateGovernanceData = () => {
  const decisions = [];
  const currentTime = Date.now();
  
  for (let i = 0; i < 50; i++) {
    const timestamp = currentTime - (i * 3600000); // Hours ago
    const threatLevel = Math.floor(Math.random() * 10) + 1;
    const urgencyLevel = Math.floor(Math.random() * 10) + 1;
    const isValid = Math.random() > 0.15; // 85% approval rate
    
    const decision = {
      id: `MARS_GOV_${Date.now() - i}`,
      timestamp,
      constitutional_validity: isValid,
      threat_level: threatLevel,
      urgency_level: urgencyLevel,
      jurisdiction: ['mars_colony', 'earth_treaty', 'emergency_protocols'][Math.floor(Math.random() * 3)],
      population_affected: Math.floor(Math.random() * 50000) + 1000,
      category: ['life_support', 'infrastructure', 'security', 'privacy', 'democratic', 'economic'][Math.floor(Math.random() * 6)],
      duration_hours: Math.random() * 168, // Up to 1 week
      zk_proof: ZKPrivacyLayer.generateZKProof(
        { constitutional_validity: isValid, jurisdiction: 'mars_colony', threat_assessment: { urgency: urgencyLevel }, proof_timestamp: timestamp },
        { sensor_details: 'classified', voting_details: 'private', affected_individuals: 'anonymous' }
      )
    };
    
    decisions.push(decision);
  }
  
  return decisions;
};

const GovernanceDashboard = () => {
  const [governanceData, setGovernanceData] = useState([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [privacyMode, setPrivacyMode] = useState(true);
  const [zkVerificationStats, setZkVerificationStats] = useState({});
  const [selectedDecision, setSelectedDecision] = useState(null);

  useEffect(() => {
    const data = generateGovernanceData();
    setGovernanceData(data);
    
    // Calculate ZK verification stats
    let totalVerificationTime = 0;
    let totalProofSize = 0;
    data.forEach(decision => {
      const verification = ZKPrivacyLayer.verifyZKProof(decision.zk_proof);
      totalVerificationTime += verification.verification_time_ms;
      totalProofSize += verification.proof_size_bytes;
    });
    
    setZkVerificationStats({
      avgVerificationTime: totalVerificationTime / data.length,
      avgProofSize: totalProofSize / data.length,
      totalDecisions: data.length,
      privacyPreserved: data.length // All decisions have privacy preserved
    });
  }, []);

  // Filter data based on timeframe
  const getFilteredData = () => {
    const now = Date.now();
    const timeframes = {
      '1h': 3600000,
      '24h': 86400000,
      '7d': 604800000,
      '30d': 2592000000
    };
    
    const cutoff = now - timeframes[selectedTimeframe];
    return governanceData.filter(decision => decision.timestamp >= cutoff);
  };

  const filteredData = getFilteredData();

  // Calculate key metrics
  const constitutionalComplianceRate = filteredData.length > 0 
    ? (filteredData.filter(d => d.constitutional_validity).length / filteredData.length) * 100 
    : 0;
  
  const citizenTrustIndex = Math.min(95, constitutionalComplianceRate + Math.random() * 10 - 5);
  
  const avgThreatLevel = filteredData.length > 0
    ? filteredData.reduce((sum, d) => sum + d.threat_level, 0) / filteredData.length
    : 0;

  // Chart data
  const complianceOverTime = filteredData
    .sort((a, b) => a.timestamp - b.timestamp)
    .map((decision, index) => ({
      time: new Date(decision.timestamp).toLocaleTimeString(),
      compliance: decision.constitutional_validity ? 100 : 0,
      threat_level: decision.threat_level,
      index
    }));

  const jurisdictionData = ['mars_colony', 'earth_treaty', 'emergency_protocols'].map(jurisdiction => ({
    name: jurisdiction.replace('_', ' ').toUpperCase(),
    value: filteredData.filter(d => d.jurisdiction === jurisdiction).length,
    compliance: filteredData.filter(d => d.jurisdiction === jurisdiction && d.constitutional_validity).length
  }));

  const categoryData = ['life_support', 'infrastructure', 'security', 'privacy', 'democratic', 'economic'].map(category => ({
    name: category.replace('_', ' ').toUpperCase(),
    decisions: filteredData.filter(d => d.category === category).length,
    avgThreat: filteredData.filter(d => d.category === category).reduce((sum, d) => sum + d.threat_level, 0) / Math.max(1, filteredData.filter(d => d.category === category).length)
  }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  const zkPrivacyMetrics = [
    { name: 'Avg Verification Time', value: `${zkVerificationStats.avgVerificationTime?.toFixed(1)}ms`, icon: Clock },
    { name: 'Avg Proof Size', value: `${zkVerificationStats.avgProofSize}B`, icon: Shield },
    { name: 'Privacy Preserved', value: `${zkVerificationStats.privacyPreserved}/${zkVerificationStats.totalDecisions}`, icon: Eye },
    { name: 'ZK Success Rate', value: '100%', icon: CheckCircle }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Shield className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              Mars Constitutional Governance Dashboard
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Eye className="w-4 h-4" />
              <span className="text-sm">Privacy Mode</span>
              <button
                onClick={() => setPrivacyMode(!privacyMode)}
                className={`w-12 h-6 rounded-full transition-colors ${privacyMode ? 'bg-green-500' : 'bg-gray-600'}`}
              >
                <div className={`w-5 h-5 bg-white rounded-full transition-transform ${privacyMode ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>
            <select 
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
        <p className="text-slate-300 text-lg">
          Real-time constitutional compliance monitoring with zero-knowledge privacy preservation
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-gradient-to-r from-green-600 to-green-500 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm">Constitutional Compliance</p>
              <p className="text-3xl font-bold text-white">{constitutionalComplianceRate.toFixed(1)}%</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-200" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-blue-600 to-blue-500 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm">Citizen Trust Index</p>
              <p className="text-3xl font-bold text-white">{citizenTrustIndex.toFixed(1)}%</p>
            </div>
            <Users className="w-8 h-8 text-blue-200" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-orange-600 to-orange-500 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-100 text-sm">Avg Threat Level</p>
              <p className="text-3xl font-bold text-white">{avgThreatLevel.toFixed(1)}/10</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-orange-200" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-purple-600 to-purple-500 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100 text-sm">Active Decisions</p>
              <p className="text-3xl font-bold text-white">{filteredData.length}</p>
            </div>
            <Globe className="w-8 h-8 text-purple-200" />
          </div>
        </div>
      </div>

      {/* ZK Privacy Metrics */}
      <div className="bg-slate-800 rounded-xl p-6 mb-8">
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <Shield className="w-5 h-5 mr-2 text-cyan-400" />
          Zero-Knowledge Privacy Layer
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {zkPrivacyMetrics.map((metric, index) => (
            <div key={index} className="bg-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <metric.icon className="w-5 h-5 text-cyan-400" />
                <span className="text-cyan-400 font-mono text-sm">{metric.value}</span>
              </div>
              <p className="text-slate-300 text-sm">{metric.name}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 p-3 bg-slate-700 rounded-lg">
          <p className="text-sm text-slate-300">
            <Zap className="w-4 h-4 inline mr-1 text-yellow-400" />
            All governance decisions use zk-SNARKs to prove constitutional compliance while preserving citizen privacy.
            Sensitive data (individual votes, sensor details, personal information) remains encrypted and unobservable.
          </p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Constitutional Compliance Over Time */}
        <div className="bg-slate-800 rounded-xl p-6">
          <h2 className="text-xl font-bold mb-4">Constitutional Compliance Timeline</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={complianceOverTime}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Line type="monotone" dataKey="compliance" stroke="#10B981" strokeWidth={2} dot={{ fill: '#10B981' }} />
              <Line type="monotone" dataKey="threat_level" stroke="#F59E0B" strokeWidth={2} dot={{ fill: '#F59E0B' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Jurisdiction Distribution */}
        <div className="bg-slate-800 rounded-xl p-6">
          <h2 className="text-xl font-bold mb-4">Decisions by Jurisdiction</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={jurisdictionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value, compliance }) => `${name}: ${value} (${((compliance/value)*100).toFixed(0)}% valid)`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {jurisdictionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Category Threat Levels */}
        <div className="bg-slate-800 rounded-xl p-6">
          <h2 className="text-xl font-bold mb-4">Threat Levels by Category</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" angle={-45} textAnchor="end" height={80} />
              <YAxis stroke="#9CA3AF" />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }} />
              <Bar dataKey="decisions" fill="#3B82F6" />
              <Bar dataKey="avgThreat" fill="#EF4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Decisions */}
        <div className="bg-slate-800 rounded-xl p-6">
          <h2 className="text-xl font-bold mb-4">Recent Constitutional Decisions</h2>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {filteredData.slice(0, 8).map((decision) => (
              <div
                key={decision.id}
                className="bg-slate-700 rounded-lg p-3 cursor-pointer hover:bg-slate-600 transition-colors"
                onClick={() => setSelectedDecision(decision)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm text-cyan-400">{decision.id}</span>
                  <div className="flex items-center space-x-2">
                    {decision.constitutional_validity ? (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    )}
                    <span className={`text-xs px-2 py-1 rounded ${decision.constitutional_validity ? 'bg-green-600' : 'bg-red-600'}`}>
                      {decision.constitutional_validity ? 'VALID' : 'BLOCKED'}
                    </span>
                  </div>
                </div>
                <div className="flex justify-between text-sm text-slate-300">
                  <span>Threat: {decision.threat_level}/10</span>
                  <span>Category: {decision.category.replace('_', ' ')}</span>
                  <span>{new Date(decision.timestamp).toLocaleTimeString()}</span>
                </div>
                {privacyMode && (
                  <div className="mt-2 text-xs text-slate-400">
                    <Shield className="w-3 h-3 inline mr-1" />
                    ZK Proof: {decision.zk_proof.zk_proof.substring(0, 16)}...
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ZK Proof Details Modal */}
      {selectedDecision && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 rounded-xl p-6 max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">Constitutional Decision Details</h3>
              <button
                onClick={() => setSelectedDecision(null)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="bg-slate-700 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Public Verification Data</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-slate-400">Decision ID:</span>
                    <p className="font-mono">{selectedDecision.id}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">Constitutional Validity:</span>
                    <p className={selectedDecision.constitutional_validity ? 'text-green-400' : 'text-red-400'}>
                      {selectedDecision.constitutional_validity ? 'VALID' : 'INVALID'}
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-400">Jurisdiction:</span>
                    <p>{selectedDecision.jurisdiction.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">Threat Level:</span>
                    <p>{selectedDecision.threat_level}/10</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-slate-700 rounded-lg p-4">
                <h4 className="font-semibold mb-2 flex items-center">
                  <Shield className="w-4 h-4 mr-2 text-cyan-400" />
                  Zero-Knowledge Proof
                </h4>
                <div className="text-sm space-y-2">
                  <div>
                    <span className="text-slate-400">Proof Hash:</span>
                    <p className="font-mono text-cyan-400">{selectedDecision.zk_proof.zk_proof}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">Verification Key:</span>
                    <p className="font-mono text-cyan-400">{selectedDecision.zk_proof.verification_key}</p>
                  </div>
                  <div className="bg-slate-600 rounded p-3 mt-3">
                    <p className="text-xs text-slate-300">
                      🔒 <strong>Privacy Preserved:</strong> Individual votes, sensor readings, and personal data 
                      are cryptographically hidden while allowing public verification of constitutional compliance.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="text-center text-slate-400 text-sm">
        <p>Mars Constitutional Governance • Real-time • Privacy-Preserving • Mathematically Verified</p>
        <p className="mt-1">Powered by zk-SNARKs • All constitutional proofs are publicly verifiable while preserving citizen privacy</p>
      </div>
    </div>
  );
};

export default GovernanceDashboard;
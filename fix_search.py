"""
Quick fix script to update search.html with dropdowns and correct API field names
"""

# Read the current basic search.html
with open('static/search.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Create the new version with dropdowns
new_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Peptide Search Explorer</title>
    <style>
        body { font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select { width: 100%; padding: 8px; box-sizing: border-box; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; margin-right: 10px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        #results, #explanation { margin-top: 20px; }
        #explanation { padding: 15px; background-color: #f0f7ff; border-radius: 4px; display: none; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .error { color: red; }
        .peptide-props { background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .explanation-text { line-height: 1.6; white-space: pre-wrap; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Peptide Search Explorer</h1>
    
    <div class="form-group">
        <label for="runId">Select Run:</label>
        <select id="runId" onchange="loadFeatures()">
            <option value="">Loading runs...</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="featureId">Select Peptide:</label>
        <select id="featureId" disabled>
            <option value="">First select a run</option>
        </select>
    </div>
    
    <button onclick="search()">🔍 Search Similar Peptides</button>
    <button id="explainBtn" onclick="explainPeptide()" disabled>🧠 Explain This Peptide</button>
    
    <div id="explanation"></div>
    <div id="results"></div>

    <script>
        let currentRunId = '';
        let currentFeatureId = '';
        const token = localStorage.getItem('token');

        window.addEventListener('DOMContentLoaded', loadRuns);

        async function loadRuns() {
            const runSelect = document.getElementById('runId');
            try {
                const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
                const response = await fetch('/runs', { headers });
                if (!response.ok) throw new Error('Failed to load runs');
                const runs = await response.json();
                if (runs.length === 0) {
                    runSelect.innerHTML = '<option value="">No runs available</option>';
                    return;
                }
                runSelect.innerHTML = '<option value="">-- Select a run --</option>' +
                    runs.map(run => `<option value="${run.run_id}">${run.run_id} (${run.n_features} features)</option>`).join('');
            } catch (e) {
                runSelect.innerHTML = `<option value="">Error: ${e.message}</option>`;
            }
        }

        async function loadFeatures() {
            const runId = document.getElementById('runId').value;
            const featureSelect = document.getElementById('featureId');
            const explainBtn = document.getElementById('explainBtn');
            featureSelect.disabled = true;
            featureSelect.innerHTML = '<option value="">Loading features...</option>';
            document.getElementById('results').innerHTML = '';
            document.getElementById('explanation').style.display = 'none';
            explainBtn.disabled = true;
            if (!runId) {
                featureSelect.innerHTML = '<option value="">First select a run</option>';
                return;
            }
            try {
                const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
                const response = await fetch(`/dashboard-data/${runId}`, { headers });
                if (!response.ok) throw new Error('Failed to load features');
                const data = await response.json();
                const features = data.data;
                if (features.length === 0) {
                    featureSelect.innerHTML = '<option value="">No features available</option>';
                    return;
                }
                featureSelect.innerHTML = '<option value="">-- Select a peptide --</option>' +
                    features.map(f => `<option value="${f.feature_id}">${f.sequence} (intensity: ${f.intensity.toExponential(2)})</option>`).join('');
                featureSelect.disabled = false;
            } catch (e) {
                featureSelect.innerHTML = `<option value="">Error: ${e.message}</option>`;
            }
        }

        async function search() {
            const runId = document.getElementById('runId').value;
            const featureId = document.getElementById('featureId').value;
            const resultsDiv = document.getElementById('results');
            const explainBtn = document.getElementById('explainBtn');
            if (!runId || !featureId) {
                resultsDiv.innerHTML = '<p class="error">Please select both a run and a peptide.</p>';
                return;
            }
            currentRunId = runId;
            currentFeatureId = featureId;
            resultsDiv.innerHTML = '<p>Searching...</p>';
            document.getElementById('explanation').style.display = 'none';
            try {
                const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
                const response = await fetch(`/peptide/search/${runId}/${featureId}`, { headers });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Search failed');
                }
                const data = await response.json();
                displayResults(data);
                explainBtn.disabled = false;
            } catch (e) {
                resultsDiv.innerHTML = `<p class="error">Error: ${e.message}</p>`;
                explainBtn.disabled = true;
            }
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            let html = `<h3>Results for ${data.query.feature_id}</h3>`;
            if (!data.neighbors || data.neighbors.length === 0) {
                html += '<p>No similar peptides found.</p>';
            } else {
                html += '<table><thead><tr><th>Feature ID</th><th>Sequence</th><th>Similarity</th></tr></thead><tbody>';
                data.neighbors.forEach(item => {
                    html += `<tr><td>${item.feature_id}</td><td>${item.peptide_sequence || 'N/A'}</td><td>${(item.similarity * 100).toFixed(1)}%</td></tr>`;
                });
                html += '</tbody></table>';
            }
            resultsDiv.innerHTML = html;
        }

        async function explainPeptide() {
            if (!currentRunId || !currentFeatureId) {
                alert('Please search for a peptide first');
                return;
            }
            const explainBtn = document.getElementById('explainBtn');
            const explanationDiv = document.getElementById('explanation');
            explainBtn.textContent = '⏳ Generating explanation...';
            explainBtn.disabled = true;
            explanationDiv.style.display = 'none';
            try {
                const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
                const response = await fetch(`/peptide/explain/${currentRunId}/${currentFeatureId}`, { headers });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Explanation failed');
                }
                const data = await response.json();
                let html = `<h3>AI Explanation</h3><div class="peptide-props"><strong>Peptide:</strong> <code>${data.sequence}</code><br><strong>Length:</strong> ${data.length} AA | <strong>Intensity:</strong> ${data.intensity.toExponential(2)} | <strong>Hydrophobicity:</strong> ${data.hydrophobicity} | <strong>Charge:</strong> ${data.charge > 0 ? '+' + data.charge : data.charge}</div><div class="explanation-text">${data.explanation}</div>`;
                if (data.neighbors && data.neighbors.length > 0) {
                    html += `<details style="margin-top: 15px;"><summary style="cursor: pointer; font-weight: bold;">Similar Peptides Used for Analysis</summary><ul style="margin-top: 10px;">`;
                    data.neighbors.forEach(n => {
                        html += `<li><code>${n.sequence}</code> (similarity: ${n.similarity_score}, intensity: ${n.intensity.toExponential(2)})</li>`;
                    });
                    html += '</ul></details>';
                }
                explanationDiv.innerHTML = html;
                explanationDiv.style.display = 'block';
            } catch (e) {
                explanationDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${e.message}</div>`;
                explanationDiv.style.display = 'block';
            } finally {
                explainBtn.textContent = '🧠 Explain This Peptide';
                explainBtn.disabled = false;
            }
        }
    </script>
</body>
</html>'''

# Write the new version
with open('static/search.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ search.html updated successfully!")
print("- Added cascading dropdowns for runs and peptides")
print("- Fixed API field names (neighbors instead of nearest_neighbors)")
print("- Added explain peptide functionality")

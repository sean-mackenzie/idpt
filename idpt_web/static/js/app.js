// IDPT Web Interface JavaScript

const API_BASE = '/api/v1';

// State
let currentJobId = null;
let pollInterval = null;

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const createTab = document.getElementById('create-tab');
const jobsTab = document.getElementById('jobs-tab');
const jobForm = document.getElementById('job-form');
const jobsList = document.getElementById('jobs-list');
const modal = document.getElementById('job-modal');
const closeModal = document.querySelector('.close');
const refreshBtn = document.getElementById('refresh-jobs');

// API Functions
async function api(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultHeaders = {};

    if (!(options.body instanceof FormData)) {
        defaultHeaders['Content-Type'] = 'application/json';
    }

    const response = await fetch(url, {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Request failed');
    }

    if (response.status === 204) {
        return null;
    }

    return response.json();
}

// Job API
const JobAPI = {
    async create(settings) {
        return api('/jobs', {
            method: 'POST',
            body: JSON.stringify({ settings }),
        });
    },

    async list(page = 1, pageSize = 10) {
        return api(`/jobs?page=${page}&page_size=${pageSize}`);
    },

    async get(jobId) {
        return api(`/jobs/${jobId}`);
    },

    async delete(jobId) {
        return api(`/jobs/${jobId}`, { method: 'DELETE' });
    },

    async uploadCalibration(jobId, files) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        return api(`/jobs/${jobId}/upload/calibration`, {
            method: 'POST',
            body: formData,
        });
    },

    async uploadTest(jobId, files) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        return api(`/jobs/${jobId}/upload/test`, {
            method: 'POST',
            body: formData,
        });
    },

    async start(jobId) {
        return api(`/jobs/${jobId}/start`, { method: 'POST' });
    },

    async getResults(jobId) {
        return api(`/jobs/${jobId}/results`);
    },
};

// Tab Navigation
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const tab = btn.dataset.tab;
        createTab.classList.toggle('active', tab === 'create');
        jobsTab.classList.toggle('active', tab === 'jobs');

        if (tab === 'jobs') {
            loadJobs();
        }
    });
});

// Form Handling
function parseFormToSettings(form) {
    const formData = new FormData(form);
    const settings = {
        name: '',
        calibration_input: {},
        test_input: {},
        calibration_processing: {},
        test_processing: {},
        output: {},
    };

    for (const [key, value] of formData.entries()) {
        const parts = key.split('.');
        if (parts.length === 1) {
            settings[key] = value;
        } else {
            const [section, field] = parts;
            if (!settings[section]) settings[section] = {};

            // Type conversion
            let parsedValue = value;
            if (field.includes('area') || field.includes('padding')) {
                parsedValue = parseInt(value, 10);
            } else if (field === 'z_step_size' || field === 'same_id_threshold' || field === 'threshold_value') {
                parsedValue = parseFloat(value);
            } else if (field === 'save_plots') {
                parsedValue = form.querySelector(`[name="${key}"]`).checked;
            } else if (field === 'xy_displacement') {
                // Parse "dx,dy" format
                if (value && value.trim()) {
                    const [dx, dy] = value.split(',').map(v => parseInt(v.trim(), 10));
                    parsedValue = [[dx, dy]];
                } else {
                    parsedValue = null;
                }
            }

            settings[section][field] = parsedValue;
        }
    }

    // Copy shared processing settings to test_processing
    const sharedFields = ['min_particle_area', 'max_particle_area', 'threshold_method', 'threshold_value', 'same_id_threshold', 'cropping_pad', 'stacks_use_raw'];
    sharedFields.forEach(field => {
        if (settings.calibration_processing[field] !== undefined && settings.test_processing[field] === undefined) {
            settings.test_processing[field] = settings.calibration_processing[field];
        }
    });

    return settings;
}

jobForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitBtn = jobForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Creating...';

    try {
        const settings = parseFormToSettings(jobForm);
        const job = await JobAPI.create(settings);
        currentJobId = job.id;
        openJobModal(job);
        showNotification('Job created successfully', 'success');
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Job';
    }
});

// Jobs List
async function loadJobs() {
    try {
        const data = await JobAPI.list(1, 20);
        renderJobs(data.jobs);
    } catch (error) {
        jobsList.innerHTML = `<div class="empty-state"><p>Error loading jobs: ${error.message}</p></div>`;
    }
}

function renderJobs(jobs) {
    if (jobs.length === 0) {
        jobsList.innerHTML = `
            <div class="empty-state">
                <p>No jobs yet</p>
                <button class="btn btn-primary" onclick="document.querySelector('[data-tab=create]').click()">Create your first job</button>
            </div>
        `;
        return;
    }

    jobsList.innerHTML = jobs.map(job => `
        <div class="job-card" data-job-id="${job.id}">
            <div class="job-card-header">
                <h3>${escapeHtml(job.name)}</h3>
                <span class="status-badge status-${job.status}">${job.status}</span>
            </div>
            <p class="job-id">${job.id}</p>
            <p class="job-date">${formatDate(job.created_at)}</p>
        </div>
    `).join('');

    // Add click handlers
    jobsList.querySelectorAll('.job-card').forEach(card => {
        card.addEventListener('click', async () => {
            const jobId = card.dataset.jobId;
            const job = await JobAPI.get(jobId);
            openJobModal(job);
        });
    });
}

refreshBtn.addEventListener('click', loadJobs);

// Modal
function openJobModal(job) {
    currentJobId = job.id;
    modal.classList.add('active');

    document.getElementById('modal-job-name').textContent = job.name;

    // Update status
    const statusDiv = document.getElementById('modal-job-status');
    statusDiv.innerHTML = `<span class="status-badge status-${job.status}">${job.status}</span>`;

    // Show/hide sections based on status
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');

    uploadSection.classList.toggle('hidden', job.status !== 'pending');
    progressSection.classList.toggle('hidden', job.status !== 'processing');
    resultsSection.classList.toggle('hidden', job.status !== 'completed');
    errorSection.classList.toggle('hidden', job.status !== 'failed');

    if (job.status === 'pending') {
        updateUploadStatus(job);
    } else if (job.status === 'processing') {
        startPolling();
        updateProgress(job.progress);
    } else if (job.status === 'completed') {
        stopPolling();
        loadResults(job.id);
    } else if (job.status === 'failed') {
        stopPolling();
        document.getElementById('error-message').textContent = job.error || 'Unknown error';
    }
}

function closeJobModal() {
    modal.classList.remove('active');
    stopPolling();
    currentJobId = null;
}

closeModal.addEventListener('click', closeJobModal);
modal.addEventListener('click', (e) => {
    if (e.target === modal) closeJobModal();
});

// Upload Handling
function updateUploadStatus(job) {
    const calibStatus = document.getElementById('calib-status');
    const testStatus = document.getElementById('test-status');
    const startBtn = document.getElementById('start-processing');

    if (job.has_calibration_images) {
        calibStatus.innerHTML = '<span class="success">Images uploaded</span>';
    } else {
        calibStatus.innerHTML = '<span>No images uploaded</span>';
    }

    if (job.has_test_images) {
        testStatus.innerHTML = '<span class="success">Images uploaded</span>';
    } else {
        testStatus.innerHTML = '<span>No images uploaded</span>';
    }

    startBtn.disabled = !(job.has_calibration_images && job.has_test_images);
}

// Drop Zone Setup
document.querySelectorAll('.drop-zone').forEach(zone => {
    const input = zone.querySelector('input[type="file"]');
    const type = zone.dataset.type;

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', async (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files);
        await uploadFiles(type, files);
    });

    input.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        await uploadFiles(type, files);
    });
});

async function uploadFiles(type, files) {
    if (!currentJobId || files.length === 0) return;

    const statusEl = document.getElementById(`${type === 'calibration' ? 'calib' : 'test'}-status`);
    statusEl.innerHTML = '<span>Uploading...</span>';

    try {
        if (type === 'calibration') {
            await JobAPI.uploadCalibration(currentJobId, files);
        } else {
            await JobAPI.uploadTest(currentJobId, files);
        }

        statusEl.innerHTML = `<span class="success">${files.length} files uploaded</span>`;

        // Refresh job status
        const job = await JobAPI.get(currentJobId);
        updateUploadStatus(job);
    } catch (error) {
        statusEl.innerHTML = `<span class="error">Error: ${error.message}</span>`;
    }
}

// Start Processing
document.getElementById('start-processing').addEventListener('click', async () => {
    if (!currentJobId) return;

    try {
        const job = await JobAPI.start(currentJobId);
        openJobModal(job);
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
});

// Progress Polling
function startPolling() {
    stopPolling();
    pollInterval = setInterval(async () => {
        if (!currentJobId) return;

        try {
            const job = await JobAPI.get(currentJobId);
            if (job.status === 'processing') {
                updateProgress(job.progress);
            } else {
                openJobModal(job);
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

function updateProgress(progress) {
    document.getElementById('progress-fill').style.width = `${progress}%`;
    document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
}

// Results
async function loadResults(jobId) {
    try {
        const results = await JobAPI.getResults(jobId);
        const resultsList = document.getElementById('results-list');

        if (results.files.length > 0) {
            resultsList.innerHTML = `<ul>${results.files.map(f => `<li>${escapeHtml(f)}</li>`).join('')}</ul>`;
        } else {
            resultsList.innerHTML = '<p>No result files available</p>';
        }
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

document.getElementById('download-results').addEventListener('click', () => {
    if (!currentJobId) return;
    window.open(`${API_BASE}/jobs/${currentJobId}/results/download`, '_blank');
});

// Delete Job
document.getElementById('delete-job').addEventListener('click', async () => {
    if (!currentJobId) return;

    if (!confirm('Are you sure you want to delete this job?')) return;

    try {
        await JobAPI.delete(currentJobId);
        closeJobModal();
        loadJobs();
        showNotification('Job deleted', 'success');
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
});

// Utilities
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleString();
}

function showNotification(message, type = 'info') {
    // Simple alert for now - could be replaced with toast notifications
    if (type === 'error') {
        alert(message);
    }
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    // Load jobs if starting on jobs tab
    if (jobsTab.classList.contains('active')) {
        loadJobs();
    }
});

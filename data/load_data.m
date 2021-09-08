function [stimuli, spikes, param] = load_data(datadir, varargin)

% function [stimuli, spikes, param] = load_data(datadir, varargin)
% Loads datafiles created by exper_scripts
%
% Input
%   datadir    - directory to load from. If empty or missing, the function will display a
%                     dialog window to choose the directory
% Output:
%   stimuli    - structure array with descriptions of all stimuli
%   spikes     - structure with basic spike information
%       spikes.timestamps - timestamps of spikes (in seconds!!!)
%       spikes.waveforms  - spike high-pass filetered waveforms (one waveform per row)
%       spikes.waveforms_raw - unfiltered waveforms straight from the data
%       spikes.clusters - cluster numbers, one for each spike timestamp
%   param      - some useful(?) data parameters, including samplerate
%
% 20080717: changed time format for spikes.timestamps from samples to
% seconds!!! Assumes the same samplerate for all daq files in a given
% recording

stimuli = [];
spikes  = [];
param   = [];

params = inputParser;
params.addParamValue('GetSpikes', true, @islogical);
params.parse(varargin{:});


if nargin<1 || isempty(datadir)
%     [stim_file, datadir] = uigetfile('*-stimuli.mat');
	datadir = uigetdir(pwd, 'Select data directory...');
	if isequal(datadir, 0) % cancelled
		return;
	end
end
%     datadir = pwd;
stim_file = dir([datadir filesep '*-stimuli.mat']);
if isempty(stim_file)
    fprintf('Can''t find any stimuli file in %s.\n', datadir);
    return;
end
stim_file = stim_file(1).name;

if isequal(stim_file, 0)
    return;
end

% temporary
expid = stim_file(1:end-12);
% stimuli file
stim_file = fullfile(datadir, stim_file);
try
    stimuli = load(stim_file);
catch
    fprintf('Can''t load %s.\n', stim_file);
    return;
end

if params.Results.GetSpikes
    % spikes file
    spike_file = dir([datadir filesep '*-tt_spikes.dat']);
    spikes = [];
    if ~isempty(spike_file)
        spike_file = fullfile(datadir, spike_file(1).name);
        try
            spikes = load('-mat', spike_file);
        catch
            fprintf('Can''t load %s.\n', spike_file);
        end
    else
        fprintf('Can''t find spike file in %s.\n', datadir);
    end

    % fprintf("sample rate %f\n", stimuli.param(1).samplerate);
    fmt=['array =' repmat(' %1.0f ',1,10) '\n'];
    fprintf(fmt, size(spikes.waveforms));
    spikes.timestamps = spikes.timestamps./stimuli.param(1).samplerate;
    
    % clusters file
    clust_file = dir([datadir filesep '*-tt_spikes.cut']);
    clusters = [];
    if ~isempty(clust_file)
        clust_file = fullfile(datadir, clust_file(1).name);
        try
            fid = fopen(clust_file);
            clusters = textscan(fid, '%n', 'CommentStyle', '%');
            clusters = clusters{:};
            fclose(fid);
        catch
            fprintf('Can''t load %s.\n', clust_file);
        end
    else
        fprintf('Can''t find cluster file in %s.\n', datadir);
    end    
end

% fix the variables for output
param   = stimuli.param;
stimuli = stimuli.stimuli;

%convert stimuli triggers from sample points to seconds
triggers = num2cell([stimuli.trigger]./param(1).samplerate);
[stimuli.trigger] = triggers{:};

% temporary
if ~isfield(param, 'expid')
    [param.expid] = deal(expid);
end
if ~isempty(spikes)
    spikes  = rmfield(spikes, 'param');
    spikes.clusters = clusters;
end

% add datafile parameter
if ~isfield(param, 'datafile')
    [param.datafile] = deal([datadir filesep param(1).expid '-data.dat']);
end

%fprintf("class of the var %s", class(param.datafile));
%fprintf("class of the var %s", param.datafile);
data_file = dir([datadir filesep '*-data.dat']);
    datacontent = [];
    if ~isempty(data_file)
        data_file = fullfile(datadir, data_file(1).name);
        try
            datacontent = load(data_file);
        catch
            fprintf('Can''t load %s.\n', data_file);
        end   
    end
%fprintf("%d", datacontent{0});
    

% add additional time parameters
if ~isfield(param, 'nseconds')
    nseconds = num2cell([param.nsamples]./[param.samplerate]);
    [param.nseconds] = nseconds{:};
end

if ~isfield(param, 'position_offset_time')
    position_offset_time = num2cell([param.position_offset]./[param.samplerate]);
    [param.position_offset_time] = position_offset_time{:};
end

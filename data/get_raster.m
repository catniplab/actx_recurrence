function [raster, time] = get_raster(varargin)

% function RASTER = get_raster('option', value, ...)
%
% returns a matrix of responses. TIMESTAMPS must be a vector of spike timestamps
% (spike positions in seconds). STIMULI is a stimuli structure, 
% Each response starts START samples before the trigger, and ends STOP samples after the trigger
%
% Input:
%   Spikes      -   spike timestamps; same as Timestamps
%   Stimuli     -   return raster of responses to these stimuli (a stimuli structure)
%   Triggers    -   trigger positions, ie positions of interest, ie stimuli
%                   triggers :-)
%   Range       -   beginning and end of each raster (in SECONDS) relative
%                   to trigger/stimuli positions; default [0 0.1]
%           NOTE: Range values are considered RELATIVE to triggers/stimuli, ie they can be negative
%                 as well. Of course, the second value must be greater (or equal) than the first one.
%   Samplerate  -   raster will be sampled at this sampling rate; default
%                   10000 Hz, ie spikes will come at 0.1 ms precision.
%
% 
% Output:
%   raster   - (sparse) matrix of responses corresponding to the specified stimuli. Each row contains a
%               response to a single stimulus. 1s correspond to spikes, 0s to nothing:-)
%               Empty if unsuccesful
%   time     - time points relative to triggers, ie time positions of each raster column
%                 relative to triggers, ie the final x-axis of a raster plot
%
% 20080718: switched to seconds from sample points; added a bunch of new
% parameters, removed some old ones
% 20080820: changed spike position rounding (line 74) from round to floor to prevent raster
% overflowing when the last spike happens to be at the very end of requested range
% 20080824: several ranges can now be used to generate a raster. The starting points, however, must
% be monotonically increasing.

params = inputParser;
params.addParamValue('Spikes',     [],      @isnumeric);
params.addParamValue('Stimuli',    [],      @isstruct);
params.addParamValue('Triggers',   [],      @isvector);
params.addParamValue('Range',      [0 0.1], @(x) isnumeric(x) & size(x, 2)==2);
params.addParamValue('Samplerate', 10000,   @(x) isnumeric(x) & numel(x)==1);

params.parse(varargin{:});

raster = [];
time   = [];

if isempty(params.Results.Triggers) && isempty(params.Results.Stimuli)
    fprintf('get_raster: No triggers supplied...\n');
    return;
end

if ~isequal(params.Results.Range(:,1), sort(params.Results.Range(:,1)))
    fprintf('get_raster: Ranges must be monotonically increasing...\n');
end

n_ranges      = size(params.Results.Range, 1);
start_samples = round(params.Results.Range(:, 1)*params.Results.Samplerate);
stop_samples  = round(params.Results.Range(:, 2)*params.Results.Samplerate);
samples       = stop_samples-start_samples;
raster_offset = cumsum(samples);
raster_offset = [0; raster_offset(1:end-1)]

if sum(samples<=0)  % at least one range is incorrect
    return;
end

if isempty(params.Results.Triggers)
    triggers = [params.Results.Stimuli.trigger];
else
    triggers = params.Results.Triggers;
end

n_triggers = numel(triggers);
fprintf("trigger and range count %d %d\n", n_triggers, n_ranges);

raster = sparse(zeros(n_triggers, sum(samples)));

for k = 1:n_triggers
  for r = 1:n_ranges
    spike_pos = params.Results.Spikes(...
                        params.Results.Spikes > (triggers(k) + params.Results.Range(r, 1)) & ...
                        params.Results.Spikes < (triggers(k) + params.Results.Range(r, 2)));
    if ~isempty(spike_pos)
        spike_pos = floor((spike_pos-triggers(k)).*params.Results.Samplerate) ...
                    - start_samples(r) + 1 ...
                    + raster_offset(r);
        raster(k, spike_pos) = 1;
    end
  end
end

time = zeros(1, sum(samples));
for r = 1:n_ranges
    time((1:samples(r))+raster_offset(r)) = params.Results.Range(r, 1): ...
                                            1/params.Results.Samplerate: ...
                                            (params.Results.Range(r, 2)-1/params.Results.Samplerate);
end
function stim = get_unique_stimuli(all_stimuli, varargin)

% function stim = unique_stimuli(all_stimuli, varargin)
%
% returns unique stimuli from ALL_STIMULI. Only 'type' and 'param' fields are considered.
%
% Input:
%   all_stimuli - Exper stimuli structure
%
% Optional parameters:
%   CompoundStimuli - true/false: consider each compound stimulus as a single one?
%                       If true, compound stimuli are considered as one stimulus;
%                       if false (default), compound stimuli are split into individual components/stimuli
%   IgnoreField  - field which is to be ignored during comparison/search
%                   string or cell array of strings (for multiple fields)                    
%
% Output:
%   stim        - structure containing only the unique stimuli from all_stimuli
%
% 20080718: Added IgnoreField parameter
% 20090323: only using FieldCountMatch when ignoring compound stimuli

stim = [];

if iscell(all_stimuli)
    all_stimuli = [all_stimuli{:}];
end

if ~isstruct(all_stimuli)
    return;
end

params = inputParser;
params.addParamValue('CompoundStimuli', false,  @islogical);
params.addParamValue('IgnoreField',     'next', @(x) ischar(x) || iscell(x));

params.parse(varargin{:});

field_count_match = ~params.Results.CompoundStimuli; % we only consider perfect matches when we don't want compound stimuli

if ~params.Results.CompoundStimuli % we don't want compound stimuli
    compound_pos = arrayfun(@(x) iscell(x.type), all_stimuli);
    % each element of compound_pos contains number of elements for each compound stimulus
    compound_stimuli = all_stimuli(compound_pos);

    all_stimuli(compound_pos) = []; % remove the compound stimuli
                                    % and keep only the type and param fields
    all_stimuli    = struct('type',  {all_stimuli.type},... 
                            'param', {all_stimuli.param});

    if ~isempty(compound_stimuli)                        
        compound_type  = {compound_stimuli.type}; % separate stimuli into the simple ones
        compound_type  = [compound_type{:}];
        compound_param = {compound_stimuli.param};
        compound_param = [compound_param{:}];
        simple_stimuli = struct('type', compound_type, 'param', compound_param);
    
        all_stimuli    = [all_stimuli simple_stimuli]'; % new array with simple stimuli only
	end
end

astim = [];

while ~isempty(all_stimuli)
    astim.type  = all_stimuli(1).type;
    astim.param = all_stimuli(1).param;
    all_stimuli(1) = [];
    stim = [stim; astim];
    match_pos = find_match(astim, all_stimuli,...
                            'IgnoreField',     params.Results.IgnoreField,...
                            'FieldCountMatch', field_count_match);
    all_stimuli(match_pos) = [];
end
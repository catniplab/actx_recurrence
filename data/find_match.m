function pos = find_match(what, where, varargin)

% function POS = find_match(WHAT, WHERE, ...)
%
% returns positions (POS) of elements of structure array WHERE which (could) match WHAT.
% Structure array WHERE is searched and each element is compared to WHAT. Succesfully matched
% elements contain all fields of WHAT with the same values...
%
% Input:
%   what    - single record (structure array with a single element) containing desired fields with
%             desired values
%   where   - structure array to search
%
%  Optional input parameters:
%   IgnoreField  - field which is to be ignored during comparison/search
%                   string or cell array of strings (for multiple fields)                    
%   PartialMatch - look for matches inside compound fields? Some fields can be compound,
%                   ie contain a cell array with different values. If PartialMatch is true,
%                   find_match will search inside these compound fields too
%                   default == false (fields must match completely)
%   FieldCountMatch - do the field counts of what and where have to match? If what contains fewer
%                   fields than the elements of what, this flag ensures that only those with exactly
%                   the same number of fields will be detected as matching
%
%
% Output:
%   pos     - logical array where 1==match, 0==nonmatch. Empty if there's an error
%
% Examples:
%   % Find positions of all 'whitenoise' stimuli in my experiment
%   stimuli = [data{:,1}];
%   mystim.type = 'whitenoise'
%   wn_pos = find_match(mystim, stimuli); % positions of whitenoise stimuli
%   n_stim = sum(wn_pos);                 % total count of all whitenoise stimuli
%   wn_stimuli = stimuli(wn_pos);         % structure with only whitenoise stimuli
%
%   % Find positions of all stimuli from the second daq/data file
%   stimuli = [data{:,1}];
%   mystim2.datafile = 2;
%   pos2 = find_match(mystim2, stimuli); % positions of stimuli from the second data file

pos = [];

if nargin<2
    return;
end

params = inputParser;
params.addParamValue('IgnoreField',     {},     @(x) (ischar(x) || iscell(x)));
params.addParamValue('PartialMatch',    false,  @islogical);
params.addParamValue('FieldCountMatch', false,  @islogical);

params.parse(varargin{:});

ignore_fields = params.Results.IgnoreField;
if ~iscell(ignore_fields)
    ignore_fields = {ignore_fields}; % the field removal below works better with cell arrays
end

is_partial_match = params.Results.PartialMatch;

if isempty(what) || isempty(where)
    return;
end

what_fields   = fieldnames(what);
n_what_fields = numel(what_fields); % remember the original number of fields
ignore_idx    = ismember(ignore_fields, what_fields);
ignore_fields = ignore_fields(ignore_idx);
if ~isempty(ignore_fields)
    what = rmfield(what, ignore_fields);
end

what_fields = fieldnames(what);
nwhere      = length(where);
pos         = true(nwhere, 1);
where       = where(:);

for field = what_fields'
    field = field{:};
    if isstruct(what.(field))
        try
            pos(pos) = pos(pos) & find_match(what.(field), {where(pos).(field)}, varargin{:});         % call myself
        catch
            % where apparently doesn't contain (field) or something else is wrong :-)
            pos = [];
            return;
        end
    else
        try
            if is_partial_match % we seek partial match, ie a field can contain multiple values 
                                % and we want to match at least one of those
                if isstruct(where)  % the following only works for fields composed of (cell arrays of) strings
                    pos = pos & ...
                      arrayfun(@(x)(isfield(x,field) && ismember({what.(field)}, x.(field))), where);
                elseif iscell(where) % we have received a call from ourselves
                    pos = pos & ...
                      cellfun(@(x)...
                                (iscell(x) && ...
                                 sum(cellfun(@(y) (isfield(y, field) && isequal(y.(field), what.(field))), x))>0) ||...
                                (isfield(x,field) && isequal(x.(field), what.(field))),...
                                where);
                end
            else
                if isstruct(where)
                    pos = pos & ...
                      arrayfun(@(x)(isfield(x,field) && isequal(x.(field), what.(field))), where);
                elseif iscell(where)
                    pos = pos & ...
                      cellfun(@(x)(isfield(x,field) && isequal(x.(field), what.(field))), where);
                end
            end
        catch
            pos = [];
            return;
        end
    end
end
       

if params.Results.FieldCountMatch && sum(pos)>0   % something was found, let's check if field counts match as well
    if isstruct(where)
        n_where_fields = arrayfun(@(x) numel(fieldnames(x)), where);
    else
        n_where_fields = cellfun(@(x) numel(fieldnames(x)), where);
    end
    pos = pos & n_where_fields == n_what_fields;
end

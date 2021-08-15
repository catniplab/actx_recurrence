function field_values = get_field_values(structure_to_search, fieldname, varargin)

% function get_field_values(structure_to_search, fieldname, varargin)
%
% searches stimuli structure (structure_to_search) and returns all unique values of the field
% fieldname
%
% Input:
%   structure_to_search - exper2 stimuli structure
%   fieldname           - name of the desired field
%
% Parameters:
%   Unique              - if true, returns only unique values. False by default.
% 
% Output:
%   field_values - array of (unique) values from the field 'fieldname'
%
% Note: for now it doesn't take compound stimuli into account
%

field_values = [];

if nargin<2 || ~isstruct(structure_to_search) || ~isvector(structure_to_search)
    return;
end

field_values = NaN(numel(structure_to_search), 1);

params = inputParser;
params.addParamValue('Unique', false, @islogical);

params.parse(varargin{:});

% first, find which elements actually contain the field

contains_field_idx = arrayfun(@(x) isfield(x, fieldname), structure_to_search);

if sum(contains_field_idx)>0    % found the field
    structure_to_search = structure_to_search(contains_field_idx);
    n_elements = numel(structure_to_search);
    % let's find out if we can concatenate the values into an array
    n_num_scalars = sum(arrayfun(@(x) isscalar(x.(fieldname)) && ~ischar(x.(fieldname)), structure_to_search));
    if isequal(n_num_scalars, n_elements)
        % we have scalar number, so we can create an array
        some_field_values = arrayfun(@(x) x.(fieldname), structure_to_search);
		field_values(contains_field_idx) = some_field_values;
    else
        % we have to return a cell array
        some_field_values = arrayfun(@(x) x.(fieldname), structure_to_search, 'UniformOutput', false);
		% so we have to convert field_values to a cell array first
		field_values = mat2cell(field_values, ones(numel(field_values), 1), 1);
		field_values(contains_field_idx) = some_field_values;
	end    
else    % didn't find the field, must go deeper
    all_fieldnames = fieldnames(structure_to_search);
    for f = all_fieldnames'
        contains_field_idx = cellfun(@(x) isfield(x, fieldname), {structure_to_search.(f{:})});
        if sum(contains_field_idx)>0
            % found the field
            structure_to_search = structure_to_search(contains_field_idx);
            n_elements = numel(structure_to_search);
            % let's find out if we can concatenate the values into an array
            n_num_scalars = sum(cellfun(@(x) isscalar(x.(fieldname)) && ~ischar(x.(fieldname)),...
                                {structure_to_search.(f{:})}));
			if isequal(n_num_scalars, n_elements)
                % we have scalar number, so we can create an array
                some_field_values = cellfun(@(x) x.(fieldname),...
                                       {structure_to_search.(f{:})});
				field_values(contains_field_idx) = some_field_values;
            else
                % we have to return a cell array
                some_field_values = cellfun(@(x) x.(fieldname),...
                                       {structure_to_search.(f{:})},...
                                       'UniformOutput', false);
				field_values = mat2cell(field_values, ones(numel(field_values), 1), 1);
				field_values(contains_field_idx) = some_field_values;
			end            
            break;
        end
    end
end

% see if we wanted only unique field values
if params.Results.Unique
    if iscell(field_values)  % for cell arrays we need to check if they're composed of strings only
                                    % otherwise unique is not going to work
		nan_idx		 = cellfun(@(x) ~ischar(x) && isnan(x), field_values);
		field_values = unique(field_values(~nan_idx));
%         string_idx	= cellfun(@(x) ischar(x), field_values);
%         if isequal(sum(string_idx), numel(field_values)); % get unique values except for NaNs
%             field_values = unique(field_values(~isnan(field_values)));
%         end
    else
        field_values = unique(field_values(~isnan(field_values)));
    end
end
 
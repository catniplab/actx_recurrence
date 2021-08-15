function contains_field_idx = contains_field(structure_to_search, fieldname)

% function contains_field(structure_to_search, fieldname)
%
% searches stimuli structure (structure_to_search) for a field named fieldname
%
% Input:
%   structure_to_search - exper2 stimuli structure
%   fieldname           - name of the desired field
% Output:
%   contains_idx - logical array containing 1s for elements containing the field, and 0s otherwise
%
% Note: for now it doesn't take compound stimuli into account

contains_field_idx = [];

if nargin<2 || ~isstruct(structure_to_search)
    return;
end

contains_field_idx = arrayfun(@(x) isfield(x, fieldname), structure_to_search);

if sum(contains_field_idx) > 0
    return;             % field was found so we can return
else                    % nothing was found, let's try one level deeper
    all_fieldnames = fieldnames(structure_to_search);
    for f = all_fieldnames'
        contains_field_idx = cellfun(@(x) isfield(x, fieldname), {structure_to_search.(f{:})});
        if sum(contains_field_idx)>0
            contains_field_idx = contains_field_idx(:);
            break;
        end
    end
end
 
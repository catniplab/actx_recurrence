function plot_raster(varargin)

%
% function plot_raster(varargin)
%
% plots a simple raster of the input data matrix
%
% Input
%   Data - (sparse) binary matrix, 1==spike, 0==nothing
%           rows correspond to trials
%           rows beginning with -1 are plotted as empty, ie as spacers
%           rows beginning with -2 are plotted as lines, ie as separators
%   Time - time vector, specifying time position for each column of Data
%   SpikeColor     - color of plotted spikes, black by default
%   SpikeWidth     - width of plotted spikes; default=1
%   SpikeHeight    - height of plotted spikes; default=1
%   SeparatorColor - color of plotted separator lines (if requested); gray
%                   by default
%	YRange - range of y-coordinates inside which the raster will be plotted
%

params = inputParser;
params.addParamValue('Data',          [],      @(x) isnumeric(x) || islogical(x));
params.addParamValue('Time',          [],      @isvector);
params.addParamValue('SpikeColor',    [0 0 0], @(x) ischar(x) || (isvector(x) && numel(x)==3));
params.addParamValue('SpikeWidth',    1,       @(x) isnumeric(x) && isscalar(x));
params.addParamValue('SpikeHeight',   1,       @(x) isnumeric(x) && isscalar(x));
params.addParamValue('SeparatorColor',[0.75 0.75 0.75], @(x) ischar(x) || (isvector(x) && numel(x)==3));
params.addParamValue('YRange',        [],      @(x) isnumeric(x));
% % % params.addParamValue('Blocks',        [],      @isvector);

params.parse(varargin{:});

if isempty(params.Results.Data)
    return;
end

[n_rows, n_cols] = size(params.Results.Data);

time = params.Results.Time(:)'; % make Time a row vector, just in case it isn't

% prepare y coordinates for plotting
if isempty(params.Results.YRange)	% nothing suppplied, let's do a simple plot
% 	y_height	 = n_rows;
	row_height	 = 1;
	spike_height = params.Results.SpikeHeight;
	y_pos		 = 0.5;
else
	y_height	 = abs(diff(params.Results.YRange));
	row_height	 = y_height/n_rows;
	spike_height = params.Results.SpikeHeight*row_height;
	y_pos		 = min(params.Results.YRange);
end	
% % % if ~isequal(numel(params.Results.Blocks), n_rows)
% % %     blocks = [];
% % % else
% % %     blocks = params.Results.Blocks;
% % % end
% % % 
% % % block_id = unique(blocks);
% % % n_blocks = numel(block_id);
% % % 
% % % if n_blocks
% % %     block_colors = [1     1     1;
% % %                     0.95  0.95  0.95];
% % %     n_block_colors = size(block_colors, 1);
% % %     block_colors = repmat(block_colors, ceil(n_blocks/n_block_colors), 1);
% % %     if isempty(params.Results.Time)
% % %         x_patch = [1 1 n_cols n_cols];
% % %     else
% % %         x_patch = [time(2) time(2) time(end) time(end)];
% % %     end
% % % end
    
hold on;

% and now plot the data themselves/itself        
% y_pos = n_rows;       % y_pos is position of the current row that we're plotting
% y_pos = 0.5;       % y_pos is position of the current row that we're plotting
    for r = 1:n_rows
        selector = params.Results.Data(r,1);
        switch selector
            case -1
                % do nothing - this is a spacer
            case -2
                % plot separator
				if isempty(time)
	                x_sep = [0; n_cols];
				else
	                x_sep = [time(1); time(end)];					
				end
                y_sep = [y_pos; y_pos] + spike_height/2;
                h_separator = line(x_sep, y_sep);
                set(h_separator, 'Color', params.Results.SeparatorColor);
            otherwise
% % %                 if n_blocks
% % %                     block_pos = find(block_id==blocks(r));
% % %                     h_patch = patch( x_patch,...
% % %                                      [0 1 1 0] + y_pos,...
% % %                                      block_colors(block_pos, :));
% % %                     set(h_patch, 'EdgeColor', 'none');
% % %                 end
                spike_pos = find(params.Results.Data(r,:));      % get spike positions
                if isempty(time)
                    raster_x  = ones(2,1)*(spike_pos);
                else
                    raster_x  = ones(2,1)*(time(spike_pos));                    
                end
                raster_y       = y_pos + [0; spike_height]*ones(size(raster_x(1,:)));
                current_raster = line(raster_x, raster_y);
                set(current_raster, 'Color',    params.Results.SpikeColor,...
                                    'LineWidth',params.Results.SpikeWidth);
        end
%         y_pos=y_pos-1;
        y_pos = y_pos + row_height;
    end

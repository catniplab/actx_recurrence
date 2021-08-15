
% OVIEDO LAB data analysis.

[stimuli,spikes,param]=load_data('Filename'); % this loads data into the workspace
% The next step is to build structures with stimuli of interest.
% There are 3 types of stimuli in our data: tones, fmsweep, whitenoise. 

% Here is an example to find all frequency sweep stimuli (fmsweep) and plot
% the raster
sweep_stim.type = 'fmsweep';                                % first specify stimulus type
sweep_stimuli= stimuli(find_match(sweep_stim, stimuli));    % find the stimulus type in the main stimuli struct in the workspace
sweepdir = get_field_values(sweep_stimuli, 'start_frequency'); % find all upward sweeps which can be identified by start_frequency  
speeds = get_field_values(sweep_stimuli, 'speed');              % get sweep speeds
% sort first by sweep direction then speed
[useless, sorted_idx] = sortrows([sweepdir speeds]);
sorted_stimuli = sweep_stimuli(sorted_idx);
[sorted_raster,time]=get_raster('Stimuli',sorted_stimuli,'Spikes',spikes.timestamps,'Range',[-0.5 2]); % make a raster plot
plot_raster('Data',sorted_raster,'Time',time,'SpikeHeight',2,'SpikeWidth',2)    % plot the raster
% getting sweep params
for i=1:length(sorted_stimuli);
    speed(i)=sorted_stimuli(i).param.speed;
end
set(gca,'YLim',[0,length(speeds)]);
set(gca,'YTick',1:length(speeds)/12:length(speeds)) 
set(gca,'YTickLabel',[25;50;75;100;125;150;-25;-50;-75;-100;-125;-150]);    %sweep speeds






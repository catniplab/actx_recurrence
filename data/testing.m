% Filename = dir("ACx_data_1/ACxCalyx/20080930-002/");
Filename = "ACx_data_1/ACxCalyx/20080930-002/";

fprintf("check 1: %s %s %d %d\n", Filename, class(Filename), size(Filename));

a = dir(strcat(Filename, filesep, '*-stimuli.mat'));
fprintf("test: %s", a);
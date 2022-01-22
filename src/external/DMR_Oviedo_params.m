1=200; %Lowest carrier frequency
f2=48000; %Maximum carrier frequency
fRD=1.5; %Maximum rate of change for RD
fFM=3; %Maximum rate of change for FM
MaxRD=4; %Maximum ripple density (cycles/oct)
MaxFM=50; %Maximum temporal modulation rate (Hz)
App=30; %Peak to peak amplitude of the ripple in dB
Fs=200e3; %Sampling rate
M=Fs*60*5; %5 minute long sounds
NCarriersPerOctave=100;
NS=ceil(NCarriersPerOctave*log2(f2/f1)); %Number of sinusoid carriers. ~100 sinusoids / octave
NB=1; %For NB=1 genarets DMR
Axis='log';
Block='n';
DF=round(Fs/1000);       %Sampling rate for envelope single is 3kHz (48kHz/16)
AmpDist='dB';
seed=789;
filename=['APR21_DMR' num2str(MaxFM) 'ctx']

% command line to generate stimuli
ripnoise(filename,f1,f2,fRD,fFM,MaxRD,MaxFM,App,M,Fs,NS,NB,Axis,Block,DF,AmpDist,seed)
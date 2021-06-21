% Code for pre-processing the data
datadir = './Data/';
subs = 14;
t = 19;

clearing = [];
thinking = [];
speaking = [];
stimuli = [];
audio = [];
face = [];
sub = [];
y =[];

for i = 1:subs
    
    %read the data files
    if i<10
        load([datadir 'Subject0' num2str(i) '/all_features_noICA.mat']);
    else
        load([datadir 'Subject' num2str(i) '/all_features_noICA.mat']);
    end
    
    %convert cell arrays to normal arrays
    l = 1;
    for j = 1:size(all_features.eeg_features.thinking_feats,2)
        if ~strcmp(all_features.prompts{1,j}, 'gnaw') && ~strcmp(all_features.prompts{1,j}, 'knew') && ~strcmp(all_features.prompts{1,j}, 'pat') && ~strcmp(all_features.prompts{1,j}, 'pot')
        if ~isempty(all_features.eeg_features.thinking_feats{1,j}) && ~isempty(all_features.eeg_features.speaking_feats{1,j}) && ~isempty(all_features.eeg_features.stimuli_feats) && ~isempty(all_features.wav_features{1,j}) && ~isempty(all_features.face_features{1,j})
            for k = 1:size(all_features.eeg_features.thinking_feats{1,j},2)/t
                clearing_temp(l,:,:,k) = all_features.eeg_features.clearing_feats{1,j}(:,(k-1)*t+1:k*t)';
                thinking_temp(l,:,:,k) = all_features.eeg_features.thinking_feats{1,j}(:,(k-1)*t+1:k*t)';
                speaking_temp(l,:,:,k) = all_features.eeg_features.speaking_feats{1,j}(:,(k-1)*t+1:k*t)';
                stimuli_temp(l,:,:,k) = all_features.eeg_features.stimuli_feats{1,j}(:,(k-1)*t+1:k*t)';
                audio_temp(l,:,k) = all_features.wav_features{1,j}(:,(k-1)*t+1:k*t)';
            end
            face_temp(l,:,:) = all_features.face_features{1,j}';
            
            %create class labels
            if strcmp(all_features.prompts{1,j}, '/iy/')
                y_temp(l,1) = 0;
            elseif strcmp(all_features.prompts{1,j}, '/uw/')
                y_temp(l,1) = 1;
            elseif strcmp(all_features.prompts{1,j}, '/piy/')
                y_temp(l,1) = 2;
            elseif strcmp(all_features.prompts{1,j}, '/tiy/')
                y_temp(l,1) = 3;
            elseif strcmp(all_features.prompts{1,j}, '/diy/')
                y_temp(l,1) = 4;
            elseif strcmp(all_features.prompts{1,j}, '/m/')
                y_temp(l,1) = 5;
            elseif strcmp(all_features.prompts{1,j}, '/n/')
                y_temp(l,1) = 6;
            end
            l = l+1;
        end
        end
    end
    
    %concatenate data
    clearing = [clearing; clearing_temp];
    thinking = [thinking; thinking_temp];
    speaking = [speaking; speaking_temp];
    stimuli = [stimuli; stimuli_temp];
    audio = [audio; audio_temp];
    face = [face; face_temp];
    sub = [sub; (i-1)*ones(size(thinking_temp,1),1)];
    y = [y; y_temp];
    
    clear clearing_temp thinking_temp speaking_temp stimuli_temp audio_temp face_temp y_temp
    
end

%remove nans
clearing(isnan(clearing)) = 0;
thinking(isnan(thinking)) = 0;
speaking(isnan(speaking)) = 0;
stimuli(isnan(stimuli)) = 0;
audio(isnan(audio)) = 0;
face(isnan(face)) = 0;

%zscore the data
clearing = zscore(clearing,0,1);
thinkking = zscore(thinking,0,1);
speaking = zscore(speaking,0,1);
stimuli = zscore(stimuli,0,1);
audio = zscore(audio,0,1);
face = zscore(face,0,1);

%save the data
save('data_all_subs.mat', 'thinking', 'speaking', 'stimuli', 'audio', 'face', 'sub', 'y', '-v7.3');      
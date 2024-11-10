% note that PPG_SQI_buf is a method from this Physionet cardiovascular toolbox https://archive.physionet.org/physiotools/physionet-cardiovascular-signal-toolbox/HEADER.shtml 



function [isGood] = isPPGGood(ppg, fs)
    isGood = false;
    % assume a highest HR of 200bpm.
    minPeakDistance = round(fs*0.3);
    
    [~, locs] = findpeaks(ppg, 'MinPeakDistance', minPeakDistance, 'MinPeakProminence', std(ppg));
    sigLen = length(ppg)/fs/60;
    tt = (0:length(ppg)-1)/fs/60;
    plot(tt, ppg)
    hold on
    plot(tt(locs), ppg(locs), 'ro')
    hold off
    
    [~, sqiMat] = PPG_SQI_buf(ppg, locs, [], round(300*fs)-1, fs);
    if (isempty(sqiMat)) % if the signal is too noisy such that no template is returned
       return 
    end
    acceptedBeats = sqiMat(:, 3) > 50;
    rrInts = diff(locs);
    rrInts = rrInts(1:length(acceptedBeats));
    goodQualityLen = sum(rrInts(acceptedBeats)) / fs / 60; % in minutes
    fprintf('Percentage of rec that is accepted %4.2f\n', goodQualityLen/sigLen*100)
    if (goodQualityLen/5 > 0.95)      
        isGood = true;
    end

end
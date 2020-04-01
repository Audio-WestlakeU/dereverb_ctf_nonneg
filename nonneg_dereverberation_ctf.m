function dereverberated_sig = nonneg_dereverberation_ctf(x,t60)
% dereverberation with CTF-based nonnegtive multichannel equalization 
% Input:
%   x: multichannel signal
%   t60: in seconds           
%   
% Author: Xiaofei Li, INRIA Grenoble Rhone-Alpes
% Copyright: Perception Team, INRIA Grenoble Rhone-Alpes
% The algorithm is described in the paper:  
% Xiaofei Li, Sharon Gannot, Laurent Girin and Radu Horaud. Multichannel Identification 
% and Nonnegative Equalization for Dereverberation and Noise Reduction based on Convolutive
% Transfer Function. IEEE/ACM Transactions on Audio, Speech and Language Processing, 26(10), 2018.
%

if nargin<2     
    t60 = 0.4;
end

if size(x,1)>size(x,2)
    x = x';
end
[micNum,~] = size(x);
mpNum = micNum*(micNum-1)/2;

fs = 16000;
winLen = 1024;    % alternatively, winLen = 768 corresponding to D50
overSamFac = 4;
fraShift = winLen/overSamFac;
win = hamming(winLen);

ofiLen = round(0.4*t60*fs/fraShift);         % length of oversampled CTF, is set to 0.4 t60
fiLen = ceil(ofiLen/overSamFac);             % length of critically sampled CTF
ofiLen = (fiLen-1)*overSamFac+1;

%% Noise PSD estimation
winCoe = zeros(fraShift,1);
for i = 1:overSamFac
    winCoe = winCoe+win((i-1)*fraShift+1:i*fraShift).^2;
end
winCoe = repmat(winCoe,[overSamFac,1]);            
awin = win./sqrt(winLen*winCoe);
nPsd = npsd_rs(x(1,:),winLen,0.75,fs);
for m = 2:micNum
    nPsd(:,:,m) = npsd_rs(x(m,:),winLen,0.75,fs);
end
nPsd = nPsd*max(awin)^2;

%% 
X = my_stft(x(1,:),winLen,fraShift,win);
for m = 2:micNum
    X(:,:,m) = my_stft(x(m,:),winLen,fraShift,win);
end
[freNum,fraNum,~] = size(X);

fraNum1 = fraNum-ofiLen+1;
X1 = zeros(freNum,fraNum1,micNum,ofiLen);
for fi = 0:ofiLen-1
    X1(:,:,:,fi+1) = X(:,ofiLen-fi:fraNum-fi,:);
end

%% CTF estimation
consVec = zeros(fiLen*micNum,1);
consVec(1:fiLen:end) = 1;

ctf = zeros(freNum,micNum,fiLen);
for fre = 1:freNum
    Xmat = zeros(mpNum*fraNum1,fiLen*micNum);
    mp = 0;
    for m1 = 1:micNum-1
        for m2 = m1+1:micNum
            mp = mp+1;
            Xmat((mp-1)*fraNum1+1:mp*fraNum1,(m1-1)*fiLen+1:m1*fiLen) = squeeze(X1(fre,:,m2,1:overSamFac:end));
            Xmat((mp-1)*fraNum1+1:mp*fraNum1,(m2-1)*fiLen+1:m2*fiLen) = -squeeze(X1(fre,:,m1,1:overSamFac:end));
        end
    end
    
    ctfFre =  (Xmat'*Xmat)\consVec;
    ctfFre = ctfFre/ctfFre(1);
    ctf(fre,:,:) = reshape(ctfFre,[fiLen,micNum]).';
end

%% l2 fit Threshold 
P = fraNum*micNum;
nPSD = mean(mean(nPsd(:,ceil(fraNum/2):end,:),2),3);
nEnergy = P*nPSD;
nStand = sqrt(P)*nPSD;
xEnergy = sum(sum(abs(X).^2,2),3);
sEnergy = max(0,xEnergy-nEnergy);
TH = nEnergy-2*nStand+0.05*sEnergy;

%% Nonnegtive multichannel equalization
S = zeros(freNum,fraNum);
for fre = 1:freNum    
    XFre = squeeze(X(fre,:,:));
    
    ctfFre = squeeze(ctf(fre,:,:));
    octfFre = zeros(micNum,ofiLen);
    octfFre(:,1:overSamFac:end) = ctfFre;
    
    Xamp = abs(XFre);
    octfamp = abs(octfFre);  
    
    %%
    C = zeros(fraNum*micNum,fraNum);    
    for m = 1:micNum
        octfm = octfamp(m,end:-1:1);
        Cm = zeros(fraNum,fraNum+ofiLen-1);
        for fra = 1:fraNum
            Cm(fra,:) = [zeros(1,fra-1),octfm,zeros(1,fraNum-fra)];
        end
        C((m-1)*fraNum+1:m*fraNum,:) = Cm(:,ofiLen:end);        
    end    
    z = Xamp(:);
    
    sini = z(1:fraNum);
    lambda = ones(fraNum,1);
    sls = opti_nonneg_l2fit(C,z,sini,lambda);
    optiCost = 1.05*norm(C*sls-z,2)^2;
    
    delta = max(TH(fre),optiCost); 
    
    lambda = [ones(fraNum,1);500];
    sl1 = opti_nonneg_bp(C,z,sls,delta,lambda);     
    
    S(fre,:) = sl1'.*exp(1i*angle(XFre(:,1).'));
    
end

dereverberated_sig = my_istft(S,fraShift,win);

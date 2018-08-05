%RBLIBSVC Automaric radial basis SVM, using nu_libsvc
%
%   W = RBLIBSVC(A,FID)

function w = rblibsvc(a,fid)

if nargin < 3, v = []; end
if nargin < 2, fid = []; end
if nargin < 1 | isempty(a)
	w = mapping(mfilename,fid);
	return
end

nu = max(testk(a,1),0.05);

d = sqrt(+distm(a));
sigmax = min(max(d)); % max: smallest furthest neighbour distance

%d = d + 1e100*eye(size(a,1));
d = d + diag(repmat(inf,[1 size(d,1)]));
sigmin = max(min(d)); % min: largest nearest neighbour distance

if sigmax == sigmin
  sigopt = sigmin;
else
  %q = (1/sigmin - 1/sigmax)/9;
  %sig = [1/sigmax:q:1/sigmin];
  %sig = ones(1,length(sig))./[1/sigmax:q:1/sigmin];
  
  % the same as above 
  %sig = 1./linspace(1/sigmax,1/sigmin,10);
	
  % start from the largest sigma
  sig = fliplr(logspace(log10(sigmin),log10(sigmax),10));
  
  w = [];
	errmin = inf;
	sigopt = 0;
	prprogress(fid,'     ')
	err  = [];
  stat = rand('state');
  for j=1:length(sig)
		s = sprintf('\b\b\b\b\b%5.0f',j);
		prprogress(fid,s);
		kernel = proxm([],'r',sig(j));
    rand('state',stat);
    err(j) = crossval(a,nu_libsvc([],kernel,nu),10,1,fid);
		if err(j) < errmin
			errmin = err(j);
			sigopt = sig(j);
		end
	end
	prprogress(fid,'\b\b\b\b\b')
end

%plot(sig,err)
fprintf('Sigma value: \n');
sig
fprintf('Error value: \n');
err
fprintf('Optimal sigma value: \n');
sigopt

w = nu_libsvc(a,proxm([],'r',sigopt),nu);


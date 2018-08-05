function hf = symmetrize_filter1(hf)

% Ensure hermetian symmetry.

% for k = 1:length(hf)
    dc_ind = (size(hf,1) + 1) / 2;
    hf(dc_ind+1:end,end,:) = conj(flipud(hf(1:dc_ind-1,end,:)));
% end
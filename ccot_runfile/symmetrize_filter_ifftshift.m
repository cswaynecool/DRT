function hf = symmetrize_filter_ifftshift(hf)

% Ensure hermetian symmetry.

    dc_ind = (size(hf,1) + 1) / 2;
    hf(dc_ind+1:end,end,:) = conj(flipud(hf(1:dc_ind-1,end,:)));
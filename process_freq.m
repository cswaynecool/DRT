function fftshift_mask=process_freq(fftshift_mask,row,col)
fftshift_mask(1,floor(col/2)+2:col )=-fftshift_mask(1,floor(col/2)+1:-1:2);
fftshift_mask(2:end,floor(col/2)+2:col )=- rot90( fftshift_mask(2:end,2:floor(col/2)+1 ),2);


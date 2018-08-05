function I = impreprocess(im)

if size(im,3)==1
    im(:,:,1,:)=im(:,:,1,:);
    im(:,:,2,:)=im(:,:,1,:);
    im(:,:,3,:)=im(:,:,1,:);
end

mean_pix = [103.939, 116.779, 123.68]; % BGR
im = permute(im, [2,1,3,4]);
im = im(:,:,3:-1:1,:);
I(:,:,1,:) = im(:,:,1,:)-mean_pix(1); % substract mean
I(:,:,2,:) = im(:,:,2,:)-mean_pix(2);
I(:,:,3,:) = im(:,:,3,:)-mean_pix(3);
end

function DrawImage(im,rect,pos,currentScaleFactor,sz,hf_mode_alpha_msf)
im    = insertShape(im, 'Rectangle', rect, 'LineWidth', 4, 'Color', 'red');
patch = get_pixels(im,pos,round(sz*currentScaleFactor),sz/4);
res   = double(real(ifft2(hf_mode_alpha_msf(:,:,1))));
figure(1),surf(res);
caxis([min(min(res)),max(max(res)),]);
hold on;
[row, col,~]=size(patch);
[X,Y]=meshgrid(1:row,1:col);
Z    =-ones(row,col)*0.002;
b    = surf(X,Y,Z,double(patch)/255);
set(b,'linestyle','none');

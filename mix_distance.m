function [mix_dis]=mix_distance(pos)
[row,col]=size(pos);
mix_xy   =zeros(row,row,col);
for i=1:col
    mix_xy(:,:,i)=repmat(pos(:,i),[1,row])-repmat(pos(:,i)',[row,1]);
end
mix_dis = sqrt(sum(mix_xy.^2,3));
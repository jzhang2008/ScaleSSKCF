function CellData2RectangleData
filename = uigetfile;
load(filename);
filelength=length(frameIndex);
RectData=zeros(filelength,4);
for i=1:filelength
    RectData(i,:)=[gtCornersAll{i}(1,1),gtCornersAll{i}(2,1),gtCornersAll{i}(1,3)-gtCornersAll{i}(1,1),gtCornersAll{i}(2,3)-gtCornersAll{i}(2,1)];
end
dlmwrite('groundtruth_rect.txt',RectData);
function label_y=binary_labeling(labels,params)
label_y=zeros(size(labels));
label_y(labels>params.theta_up)=1;
label_y(labels<params.theta_low)=-1;
end

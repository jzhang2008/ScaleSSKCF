function [K_num,part_sz,part_pos]=constructed_layout(init_target_sz,Aspect_ratio,pos,params)
   if Aspect_ratio<=0.6
        K_num        = 3;
        part_layout  = [1,3];
        part_sz      = ceil(init_target_sz./part_layout);
        idx_x        = [0 0 0 ];
        idx_y        = [-1 0 1];
        pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
   elseif Aspect_ratio>1.6
        K_num        = 3;
        part_layout  = [3,1];
        part_sz      = ceil(init_target_sz./part_layout);
        idx_x        = [-1 0 1]';
        idx_y        = [0 0 0 ]';
        pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
   else
            if params.pflag  == 0
               K_num            = 4;
%                part_layout      = [2,2];
               part_sz          = ceil(init_target_sz.*2/3);
               idx_x            = [-0.25 0.25]'*[1 1 ];
               idx_y            = [1 1 ]'* [-0.25 0.25];
               pos_vec          = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
            else
               K_num            = 4;
               part_sz          = ceil(init_target_sz*2/3);
               idx_x            = [-0.25 0.25]'*[1 1 ];
               idx_y            = [1 1 ]'* [-0.25 0.25];
               pos_vec          = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz)); 
            end
   end
%      if Aspect_ratio<=1
%         K_num        = 2;
%         part_layout  = [1,2];
%         part_sz          = ceil(init_target_sz./part_layout);
%         idx_x        = [0 0 ];
%         idx_y        = [-0.5 0.5];
%         pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
%      else
%         K_num        = 2;
%         part_layout  = [2,1];
%         part_sz      = ceil(init_target_sz./part_layout);
%         idx_x        = [-0.5 0.5]';
%         idx_y        = [0 0 ]';
%         pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
%      end
%     else
%         K_num        = 4;
%         part_layout  = [2,2];
%         part_sz      = ceil(init_target_sz./part_layout);
%         idx_x        = [-0.5 0.5]'*[1 1 ];
%         idx_y        = [1 1 ]'* [-0.5 0.5];
%         pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
%     end
% end
part_pos =ones(K_num,1)*pos+pos_vec;
end
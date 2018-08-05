function [patch_mask]=get_binary_patch_mask(binary_mask,frag_num)

for block_id=1:length(binary_mask)
     [x y]=find(binary_mask{block_id}>0);
     x_min=min(x(:)); x_max=max(x(:)); y_min=min(y(:)); y_max=max(y(:));
     height=x_max-x_min+1; width=y_max-y_min+1;

     mod_height=mod(height,sqrt(frag_num)); mod_width=mod(width,sqrt(frag_num));
     patch_mask{block_id}=zeros(size(binary_mask{block_id}));
     
     xx_min(1)=x_min; yy_min(1)=y_min;
     for i=2:sqrt(frag_num)
         if i<mod_height+2
             xx_min(i)=xx_min(i-1)+floor(height/sqrt(frag_num))+1;
         else
             xx_min(i)=xx_min(i-1)+floor(height/sqrt(frag_num));
         end
     end
     xx_min(end+1)=x_max+1;
     for i=2:sqrt(frag_num)
         if i<mod_width+2
             yy_min(i)=yy_min(i-1)+floor(width/sqrt(frag_num))+1;
         else
             yy_min(i)=yy_min(i-1)+floor(width/sqrt(frag_num));
         end
     end
     yy_min(end+1)=y_max+1;
     
     for i=1:sqrt(frag_num)
         for j=1:sqrt(frag_num)
                 layer_index=(i-1)*3+j;
                   patch_mask{block_id}(xx_min(i) :xx_min(i+1)-1   ,  yy_min(j) :yy_min(j+1)-1, layer_index)=1;
         end
     end
    clear xx_min; clear yy_min;
end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%================================================================
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
%===============================================================
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%===========================
% Setup some useful variables
#===========================
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%=================================================================
%add column of 1 in X matrix
 
  X= [ones(m,1) X];

% compute the z(2) and h(activation unit) for the second layer 

  z2= X* Theta1';  % 5000 by 25 matrix 
  z2_1= [ones(size(z2,1),1) z2];  % add one
  h2= sigmoid(z2);   % 5000 by 25 matrix

% compute the z(3) and h(3)the activation unit of the output layer 
  
  h2_1= [ones(size(h2,1),1) h2] ; %  add intercept for layer two 

  z3= h2_1 * Theta2';  % 5000 by 10 matrix 
  h3= sigmoid(z3);   % 5000 by 10 matrix


  %===============================
  % the cost of the neural network 
  %===============================

  cost_label=0; % initialize the cost of each output label as zero

  for j= 1:num_labels
      
     for i= 1: m
         
	if y(i)==j
           % when y(i)==j,i.e the value of y(i) equal to the level j,  then we sign 
           % y(i)=1 ,otherwise y(i)=0, 
           %this converts the y vector into binomial of 1 and zero
           % cost_label= -1*log(h3(i,j))- (1-1)log(1-h3(i,j))   
          
          cost_label= cost_label+ 1*log(h3(i,j));
        
        else 
           cost_label= cost_label + log(1-h3(i,j)); % when y=0
	  
        end;
     end
  end
 
J= -(1/m) * cost_label ;
        
%========================================================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
   %——————————————————————————————————————————————————————————————
   % compute the sigma of the output layer(sigma3), sigma3= h3-y
   %——————————————————————————————————————————————————————————————
    sigma3= zeros(m,num_labels);  % the error of output layer is a 5000 X 10 matrix
    size(sigma3);
    for j= 1:num_labels 
     
       for i= 1:m 
        
        if y(i)==j
          sigma3(i,j) = h3(i,j)-1;
        else
         sigma3(i,j) = h3(i,j); 
        end 
       end
    end
   % _______________________________________________________
   %compute sigma2 using sigmoid gradient function, and Theta _grad1
   %sigma2 = Theta2'*sigma3.* sigmoidGradient(z2);
   %________________________________________________________

    sigma2= (sigma3 *Theta2).* sigmoidGradient(z2_1); % 5000 by 26 matrix
    size(sigma2);    % to debugg the size of sigma2
    sum_hs1=0;

    sigma2= sigma2(:,2:end);  % remove the sigma0 value from the matrix
    
   %for i = 1:input_layer_size +1 
     
     %  for j= 1:hidden_layer_size  
       
      %     for t= 1:m
      %         sum_hs1= sum_hs1 + X(t,i) *sigma2(t,j);
      %     end
     
      %     Theta1_grad(j,i)= 1/m*sum_hs1; 
      %     sum_hs1=0;  % initialize/restart/ the sum as zero for next iteration 
       
     %  end
    
   % end 
   %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<
   %Alternative to the above for loop, we can use vectorized form as: 
  
     Theta1_grad= 1/m * sigma2' * X;
   %>>>>>>>>>><><><><><><><<<<<<<<<<<<<<<<<<<<<<<


   %________________________________________
   % compute gradients: Theta_grad2 
   %________________________________________
    
    sum_hs=0;

    for i= 1:hidden_layer_size +1 
       for j = 1: num_labels
         for t= 1:m       
            sum_hs= sum_hs + h2_1(t,i) * sigma3(t,j);

         end
         
           Theta2_grad (j,i)= 1/m * sum_hs;
           sum_hs = 0;    % initialize the sum as zero for next iteration 
       end
    end 

%=====================================================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
  %———————————————————————————————————
  % cost function with regularization 
  %___________________________________

  Theta1_no_intr= Theta1(: , 2:end);
  Theta2_no_intr= Theta2(:, 2:end);

  J= J + lambda/(2*m) * (sum(Theta1_no_intr(:).^2) + sum(Theta2_no_intr(:).^2));
 
  % __________________________________________
  %Gradients using regularization 
  %___________________________________________
   
  % make the intercept zero for theta1 and theta2

   Theta1_int= [zeros(size(Theta1_no_intr, 1),1) Theta1_no_intr]; 
   Theta2_int= [zeros(size(Theta2_no_intr, 1),1) Theta2_no_intr]; 
  
   Theta1_grad = Theta1_grad + (lambda/m) * Theta1_int;
   Theta2_grad = Theta2_grad + (lambda/m) * Theta2_int; 




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

i = size(z,1);
j = size(z,2);

for i=1:i,
	for j=1:j,		
		g(i,j) = 1 / ( 1 + e ^ (z(i,j)*-1));
	end;
end;

% =============================================================

end

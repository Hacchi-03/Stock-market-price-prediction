pattern=csvread('traincode.csv');
fid = fopen('wih.dat','w'); % Weights stored of inputhidden layer
fid1 = fopen('whj.dat','w'); % Output stored of hiddenoutput layer
alpha =0.9; % Momentum
%Convergence is made faster if a momentum factor is added to the weight updation process.
eta = 0.8; % Learning rate 
tol = 0.001; % Error tolerance
Q = 249; % Total no. of the patterns to be input
n = 3; q = 2; p = 1; % Architecture
% Initializing the values and weights
Wih = 2 * rand(n,q) - 1; % Input-hidden random weight matrix
Whj = 2 * rand(q,p) - 1; % Hidden-output random weight matrix
DeltaWih = zeros(n,q); % Weight change matrices
DeltaWhj = zeros(q,p); % matrix of qxp of zeros
DeltaWihOld = zeros(n,q);
DeltaWhjOld = zeros(q,p);
Si = [pattern(:,1:3)]; % Input signals
D = pattern(:,4); % Desired values
Sh = zeros(1,q); % Hidden neuron signals loop
i=0;
itt=0;
Sy = zeros(1,p); % Output neuron signals
deltaO = zeros(1,p); % Error-slope product at output
deltaH = zeros(1,q); % Error-slope product at hidden
sumerror = 2*tol; % To get in to the
figure;
%x=linespace(0,1);
tr=plot(D);
tr.Color=[0 0 0.5];
tr.Marker='o';
%hold on;
% Training BPA network
while sumerror>tol && itt<3000 % Iterate(Stops when error tolerance = 0.001 or when iteration reaches 20000 %#ok<CTPCT> 
sumerror = 0;  
for k = 1:Q % for loop of input data (Q=99 times)
 Zh = Si(k,:) * Wih; % Hidden activations
 Sh = [1./(1 + exp(-Zh))]; % Binary sigmoid function Hidden signals
 Yj = Sh * Whj; % Output activations
 Sy = 1./(1 + exp(-Yj)); % Binary sigmoid function Output signals
 Ek = D(k) - Sy; % Error vector
 deltaO = Ek .* Sy .* (1 - Sy); % Delta output
 for h = 1:q % Delta W:hidden-output
 DeltaWhj(h,:) = deltaO * Sh(h);
 end
 for h = 2:q % Delta hidden
 deltaH(h) = (deltaO * Whj(h,:)') * Sh(h) * (1-Sh(h));
 end
 for i = 1:n % Delta W: inputhidden
 DeltaWih(i,:) = deltaH(q:q) * Si(k,i);
 end
 Wih = Wih + eta * DeltaWih + alpha * DeltaWihOld;
 Whj = Whj + eta * DeltaWhj + alpha * DeltaWhjOld;
 DeltaWihOld = DeltaWih; % Updateweights(or)Store changes
 DeltaWhjOld = DeltaWhj;
 sumerror = sumerror + sum(Ek.^2); % Compute error
end
Iteration = itt
sumerror % Print epoch error
itt=itt+1;
end
Wih
Whj
fprintf(fid,'%12.8f\n',Wih);
fprintf(fid1,'%12.8f\n',Whj);
status = fclose(fid);
status = fclose(fid1);
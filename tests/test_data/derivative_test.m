% Path update
addpath(genpath('~/repos/k-wave-matlab'))
plot = false;

% Scalar test signal
a = zeros(16,16);
a(8,:) = 1;
a(:,8) = 1;

% Differential operators
da_dx = fft_derivative(a, 1);
da_dy = fft_derivative(a, 2);
nabla_a = fft_laplacian(a);

% Save data for testing
writematrix(a, 'CROSS_IMG.txt', 'delimiter', ',');
writematrix(da_dx, 'CROSS_IMG_FOURIER_DX.txt', 'delimiter', ',');
writematrix(da_dy, 'CROSS_IMG_FOURIER_DY.txt', 'delimiter', ',');
writematrix(nabla_a, 'CROSS_IMG_FOURIER_LAPLACIAN.txt', 'delimiter', ',');

% Vector test signal
b = zeros(16,16,2);
b(8,:,1) = 1;
b(:,8,1) = 1;
b(4,:,2) = -1;
b(:,4,2) = -1;

% Differential operators
nabla_dag_u = fft_diag_jacobian(b);

% Save data for testing
writematrix(b, 'CROSS_IMG_VEC.txt', 'delimiter', ',');
writematrix(nabla_dag_u, 'CROSS_IMG_VEC_FOURIER_DAG.txt', 'delimiter', ',');

if plot
  figure;
  subplot(2,2,1);
  imagesc(b(:,:,1))
  subplot(2,2,2);
  imagesc(b(:,:,2))
  subplot(2,2,3);
  imagesc(nabla_dag_u(:,:,1))
  subplot(2,2,4);
  imagesc(nabla_dag_u(:,:,2))
end


function nabla_dag_u = fft_diag_jacobian(u)
  %FFT_DIAG_JACOBIAN Compute the diagonal Jacobian of a vector field
  %   using the FFT.
  %
  %   nabla_dag_u = fft_diag_jacobian(u)
  %
  %   Input:
  %     u: vector field. The last axis is the vector dimension.
  %
  %   Output:
  %     nabla_dag_u: diagonal Jacobian of u

  % Parsing input
  arguments
    u (:,:,:) double
  end

  nabla_dag_u = zeros(size(u),'like',u);

  for vec_dim = 1:size(u,3)
    % Compute the derivative in the Fourier domain
    dan_dn = fft_derivative(squeeze(u(:,:,vec_dim)), vec_dim);
    nabla_dag_u(:,:,vec_dim) = dan_dn;
  end
end

function du = fft_derivative(u, axis, order)
  %FFT_DERIVATIVE Compute the action of the derivative operator
  %   using the FFT.
  %
  %   du = fft_derivative(u, axis, order)
  %
  %   Inputs:
  %     u: the input array (2D)
  %     axis: the axis along which to take the derivative
  %     order: the order of the derivative
  %
  %   Outputs:
  %     du: the derivative of u

  % Parsing inputs
  arguments
    u (:,:) double
    axis (1,1) {mustBeMember(axis, [1 2])} = 1
    order (1,1) {mustBeInteger, mustBePositive} = 1
  end

  % Getting filter
  kgrid = kWaveGrid(size(u,1), 1.0, size(u,2), 1.0);
  if axis == 1
    filter = kgrid.kx;
  elseif axis == 2
    filter = kgrid.ky;
  else
    error('Invalid axis')
  end

  % Unwrapping filter
  filter = 1i*ifftshift(filter);

  % Apply via fft
  du = ifft2(fft2(u) .* filter.^order);

  % Cast to real if input was real
  if isreal(u)
    du = real(du);
  end

end

function Lu = fft_laplacian(u)
  %FFT_LAPLACIAN Compute the action of the laplacian operator
  %   using the FFT.
  %
  %   Lu = fft_laplacian(u)
  %
  %   Inputs:
  %     u: the input array (2D)
  %
  %   Outputs:
  %     Lu: the laplacian of u

  % Parsing inputs
  arguments
    u (:,:) double
  end

  % Getting filter
  kgrid = kWaveGrid(size(u,1), 1.0, size(u,2), 1.0);
  filter = -(kgrid.kx.^2 + kgrid.ky.^2);

  % Unwrapping filter
  filter = ifftshift(filter);

  % Apply via fft
  Lu = ifft2(fft2(u) .* filter);

  % Cast to real if input was real
  if isreal(u)
    Lu = real(Lu);
  end

end

n = 1024;
r = 100;
p = rand([n n])*2*pi;
mask = rand([n n]) > 0.5;

tic
for i=1:r
    field = mask .* exp(1i*p);
end
toc
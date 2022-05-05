function RMSE = RMSE2(a,b)

aNorm = norm(a(:));
abNorm = norm(a(:)-b(:));
RMSE = abNorm/aNorm;

end
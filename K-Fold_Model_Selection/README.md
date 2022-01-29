# K Fold Cross Validation

K Fold Cross Validation datayı ezberlemeyi önleyen yöntemlerden birisidir.
Elimizde bir data olduğu zaman data'yı train ve test olarak 2'ye ayırırız.
İlk başlangıçta test verilerine bakmadan k değerime göre train verilerimi bölerim. Örneğin k değeri 3 olsun. Bu durumda train verilerimi 3 eşit parçaya bölerim. Herdefasında 2 parçayı train 1 parçayı da validation olarak alıp test verilerini kullanmadan trainler içinde test yapıp accuracy değerlerini buluyorum. Bu accuracy değerlerini toplayıp ortalamasını buluyorum ve modelimin accuracy değerini hesaplamış oluyorum. 
